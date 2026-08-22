package cc.pipo.cutter

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ceil

/**
 * 视频 → 单声道 float PCM，对应桌面版的 `ffmpeg -vn -ac 1 -ar N -f f32le`。
 *
 * 检测和嵌入要两个采样率：检测用 16kHz（config.audio.sr），
 * PANNs 用 32kHz。手机录像基本都是 48kHz，所以两条都要重采样。
 *
 * 尽量让解码器直接吐 float（API 24 起支持 ENCODING_PCM_FLOAT）——
 * 退回 16 位整数的话，量化会带来约 3e-5 的相对误差。对谱通量这种
 * 看相对突变的量影响很小，但能省就省。
 *
 * 内存：为什么解两遍
 * ------------------
 * 原来是「整段按**原始采样率**攒成一堆小数组，再另开一个同样大的数组
 * 拼起来，最后才降采样」。12 分钟 48kHz 单声道 = 139MB，拼接时要同时
 * 拿着两份，峰值 279MB。一台真实用户的手机就死在这里：
 *
 *     Failed to allocate a 139739088 byte allocation
 *     with 25100288 free bytes and 55MB until OOM
 *
 * 而目标采样率下只有 46MB（16kHz）—— **那 279MB 全是中转开销。**
 * 手上的测试素材都是 14-60 秒（最多 11MB），所以从没触发过。
 *
 * 现在：第一遍只数采样数，什么都不存；第二遍按算出来的精确长度
 * **一次分配**，边解边重采样直接写进去。峰值就是结果本身，没有中转，
 * 也没有「先大后小再拷一次」的瞬时双份。
 *
 * 代价是解码跑两遍（12 分钟素材多花几秒）。换来的是**内存精确可预测** ——
 * 这一条很重要：上层的时长上限守卫要靠它算，估不准就只能保守到没法用，
 * 或者乐观到又被系统杀掉。
 */
object AudioDecode {

    class NoAudioTrack : Exception("这段视频没有音轨")

    /** 解出整段音频。返回单声道 float，采样率为 [targetRate]。 */
    fun decode(path: String, targetRate: Int): FloatArray =
        decode(targetRate) { it.setDataSource(path) }

    /**
     * 相册选出来的是 content:// URI，不是文件路径 —— 而且大多数情况下
     * **拿不到真实路径**（作用域存储）。MediaExtractor 支持直接吃 URI，
     * 所以不要试图去反解路径，那条路在新系统上会时灵时不灵。
     */
    fun decode(context: Context, uri: Uri, targetRate: Int): FloatArray =
        decode(targetRate) { it.setDataSource(context, uri, null) }

    /** 解码器报出来的音轨参数。 */
    private class Info(val srcRate: Int, var channels: Int)

    private fun decode(targetRate: Int, open: (MediaExtractor) -> Unit): FloatArray {
        // 第一遍：只数单声道采样数。不存任何 PCM。
        var count = 0L
        val info = pump(open) { _, n -> count += n }
        val srcRate = info.srcRate

        if (srcRate == targetRate) {
            // 同采样率：没有重采样，长度就是采样数
            val out = FloatArray(count.toInt())
            var off = 0
            pump(open) { mono, n -> System.arraycopy(mono, 0, out, off, n); off += n }
            return out
        }

        // 输出长度和 Resampler.resample 的算法完全一致，不能各算各的
        val g = gcd(srcRate, targetRate)
        val up = targetRate / g
        val down = srcRate / g
        val outLen = ceil(count.toDouble() * up / down).toInt()

        val out = FloatArray(outLen)
        var w = 0
        val stream = Resampler.Stream(srcRate, targetRate)
        pump(open) { mono, n ->
            stream.feed(mono, n) { v -> if (w < outLen) out[w++] = v }
        }
        stream.finish { v -> if (w < outLen) out[w++] = v }
        return out
    }

    private fun gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)

    /**
     * 跑一遍解码，把每一块**已经混成单声道**的 PCM 交给 [onMono]。
     *
     * 传给回调的是一个复用的暂存数组加有效长度 —— 回调必须当场用掉，
     * 不能存下来。这正是不再攒 chunks 的关键：任何时刻只有一块在内存里。
     *
     * 逐块混单声道和「先拼再混」等价：解码器吐出来的每一块都是整帧，
     * 不会把一个采样的左右声道劈到两块里。
     */
    private inline fun pump(
        open: (MediaExtractor) -> Unit,
        onMono: (FloatArray, Int) -> Unit,
    ): Info {
        val ex = MediaExtractor()
        open(ex)
        var track = -1
        var format: MediaFormat? = null
        for (i in 0 until ex.trackCount) {
            val f = ex.getTrackFormat(i)
            if (f.getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true) {
                track = i; format = f; break
            }
        }
        if (track < 0 || format == null) { ex.release(); throw NoAudioTrack() }
        ex.selectTrack(track)

        val srcRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        val info = Info(srcRate, format.getInteger(MediaFormat.KEY_CHANNEL_COUNT))
        // 请求 float 输出；解码器不支持时会忽略，下面按实际格式读
        format.setInteger(MediaFormat.KEY_PCM_ENCODING, 4 /* ENCODING_PCM_FLOAT */)

        val codec = MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME)!!)
        codec.configure(format, null, null, 0)
        codec.start()

        val bi = MediaCodec.BufferInfo()
        var sawInputEos = false
        var sawOutputEos = false
        var isFloat = false
        var checkedFormat = false
        // 复用的暂存：交织的一块，和混完单声道的一块。按需长大，不逐块新建。
        var inter = FloatArray(8192)
        var mono = FloatArray(8192)

        try {
            while (!sawOutputEos) {
                if (!sawInputEos) {
                    val inIdx = codec.dequeueInputBuffer(10_000)
                    if (inIdx >= 0) {
                        val buf = codec.getInputBuffer(inIdx)!!
                        val n = ex.readSampleData(buf, 0)
                        if (n < 0) {
                            codec.queueInputBuffer(inIdx, 0, 0, 0,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            sawInputEos = true
                        } else {
                            codec.queueInputBuffer(inIdx, 0, n, ex.sampleTime, 0)
                            ex.advance()
                        }
                    }
                }
                val outIdx = codec.dequeueOutputBuffer(bi, 10_000)
                when {
                    outIdx >= 0 -> {
                        if (!checkedFormat) {
                            val of = codec.outputFormat
                            isFloat = of.containsKey(MediaFormat.KEY_PCM_ENCODING) &&
                                of.getInteger(MediaFormat.KEY_PCM_ENCODING) == 4
                            if (of.containsKey(MediaFormat.KEY_CHANNEL_COUNT)) {
                                info.channels = of.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                            }
                            checkedFormat = true
                        }
                        if (bi.size > 0) {
                            val buf = codec.getOutputBuffer(outIdx)!!
                            buf.position(bi.offset)
                            buf.limit(bi.offset + bi.size)
                            val got = readPcm(buf, isFloat, inter)
                            if (got > inter.size) {          // 装不下，长大再读一次
                                inter = FloatArray(got)
                                buf.position(bi.offset)
                                buf.limit(bi.offset + bi.size)
                                readPcm(buf, isFloat, inter)
                            }
                            val ch = info.channels
                            val frames = if (ch <= 1) got else got / ch
                            if (mono.size < frames) mono = FloatArray(frames)
                            if (ch <= 1) {
                                System.arraycopy(inter, 0, mono, 0, frames)
                            } else {
                                for (i in 0 until frames) {
                                    var s = 0.0
                                    for (c in 0 until ch) s += inter[i * ch + c]
                                    mono[i] = (s / ch).toFloat()
                                }
                            }
                            if (frames > 0) onMono(mono, frames)
                        }
                        codec.releaseOutputBuffer(outIdx, false)
                        if (bi.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                            sawOutputEos = true
                        }
                    }
                    outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> checkedFormat = false
                }
            }
        } finally {
            runCatching { codec.stop() }
            codec.release()
            ex.release()
        }
        return info
    }

    /**
     * 把一块 PCM 读进 [dst]，返回**实际的采样数**。
     * 返回值大于 dst.size 时表示没装下，调用方需要扩容重读。
     */
    private fun readPcm(buf: ByteBuffer, isFloat: Boolean, dst: FloatArray): Int {
        val b = buf.order(ByteOrder.nativeOrder())
        return if (isFloat) {
            val fb = b.asFloatBuffer()
            val n = fb.remaining()
            if (n <= dst.size) for (i in 0 until n) dst[i] = fb.get()
            n
        } else {
            val sb = b.asShortBuffer()
            val n = sb.remaining()
            // 除以 32768 而不是 32767：和 ffmpeg 的 s16→flt 换算一致
            if (n <= dst.size) for (i in 0 until n) dst[i] = sb.get() / 32768.0f
            n
        }
    }
}
