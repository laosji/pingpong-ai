package cc.pipo.cutter

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 视频 → 单声道 float PCM，对应桌面版的 `ffmpeg -vn -ac 1 -ar N -f f32le`。
 *
 * 检测和嵌入要两个采样率：检测用 16kHz（config.audio.sr），
 * PANNs 用 32kHz。手机录像基本都是 48kHz，所以两条都要重采样。
 *
 * 尽量让解码器直接吐 float（API 24 起支持 ENCODING_PCM_FLOAT）——
 * 退回 16 位整数的话，量化会带来约 3e-5 的相对误差。对谱通量这种
 * 看相对突变的量影响很小，但能省就省。
 */
object AudioDecode {

    class NoAudioTrack : Exception("这段视频没有音轨")

    /**
     * 解出整段音频。返回单声道 float，采样率为 [targetRate]。
     *
     * 一小时 48kHz 的素材解到 16kHz 是 5760 万个 float = 230MB —— 在手机上
     * 这个量级是危险的。调用方要么分段处理，要么先确认时长。
     * 这里不做分块是因为检测本身需要完整序列（自适应阈值按 2 秒块统计，
     * 跨块边界要连续）。
     */
    fun decode(path: String, targetRate: Int): FloatArray =
        decode(targetRate) { it.setDataSource(path) }

    /**
     * 相册选出来的是 content:// URI，不是文件路径 —— 而且大多数情况下
     * **拿不到真实路径**（作用域存储）。MediaExtractor 支持直接吃 URI，
     * 所以不要试图去反解路径，那条路在新系统上会时灵时不灵。
     */
    fun decode(context: Context, uri: Uri, targetRate: Int): FloatArray =
        decode(targetRate) { it.setDataSource(context, uri, null) }

    private fun decode(targetRate: Int, open: (MediaExtractor) -> Unit): FloatArray {
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
        var channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        // 请求 float 输出；解码器不支持时会忽略，下面按实际格式读
        format.setInteger(MediaFormat.KEY_PCM_ENCODING, 4 /* ENCODING_PCM_FLOAT */)

        val codec = MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME)!!)
        codec.configure(format, null, null, 0)
        codec.start()

        val chunks = ArrayList<FloatArray>()
        var total = 0
        val info = MediaCodec.BufferInfo()
        var sawInputEos = false
        var sawOutputEos = false
        var isFloat = false
        var checkedFormat = false

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
                val outIdx = codec.dequeueOutputBuffer(info, 10_000)
                when {
                    outIdx >= 0 -> {
                        if (!checkedFormat) {
                            val of = codec.outputFormat
                            isFloat = of.containsKey(MediaFormat.KEY_PCM_ENCODING) &&
                                    of.getInteger(MediaFormat.KEY_PCM_ENCODING) == 4
                            if (of.containsKey(MediaFormat.KEY_CHANNEL_COUNT)) {
                                channels = of.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                            }
                            checkedFormat = true
                        }
                        if (info.size > 0) {
                            val buf = codec.getOutputBuffer(outIdx)!!
                            buf.position(info.offset)
                            buf.limit(info.offset + info.size)
                            val f = readPcm(buf, isFloat)
                            chunks.add(f); total += f.size
                        }
                        codec.releaseOutputBuffer(outIdx, false)
                        if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
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

        var pcm = FloatArray(total)
        var off = 0
        for (c in chunks) { System.arraycopy(c, 0, pcm, off, c.size); off += c.size }
        pcm = Resampler.toMono(pcm, channels)
        return Resampler.resample(pcm, srcRate, targetRate)
    }

    private fun readPcm(buf: ByteBuffer, isFloat: Boolean): FloatArray {
        val b = buf.order(ByteOrder.nativeOrder())
        return if (isFloat) {
            val fb = b.asFloatBuffer()
            FloatArray(fb.remaining()) { fb.get() }
        } else {
            val sb = b.asShortBuffer()
            // 除以 32768 而不是 32767：和 ffmpeg 的 s16→flt 换算一致
            FloatArray(sb.remaining()) { sb.get() / 32768.0f }
        }
    }
}
