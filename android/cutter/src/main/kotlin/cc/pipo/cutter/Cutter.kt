package cc.pipo.cutter

import android.content.Context
import android.net.Uri
import android.os.Handler
import android.os.HandlerThread
import androidx.media3.common.MediaItem
import androidx.media3.common.MimeTypes
import androidx.media3.common.util.UnstableApi
import androidx.media3.effect.Presentation
import androidx.media3.transformer.Composition
import androidx.media3.transformer.EditedMediaItem
import androidx.media3.transformer.EditedMediaItemSequence
import androidx.media3.transformer.Effects
import androidx.media3.transformer.ExportException
import androidx.media3.transformer.ExportResult
import androidx.media3.transformer.ProgressHolder
import androidx.media3.transformer.Transformer
import com.google.common.collect.ImmutableList
import java.io.File
import kotlin.math.roundToLong

/**
 * 把若干时间区间剪出来接成一条成片。
 *
 * 为什么必须重编码，不能用 MediaExtractor + MediaMuxer 做流拷贝
 * -------------------------------------------------------------
 * 流拷贝的切点只能落在关键帧上。实测 8 段真实手机录像的关键帧间隔：
 * 6 段是 **1.00 秒**，2 段是 **3.00 秒**。也就是说片段最多会提前 3 秒开始，
 * 而我们的回合本身只有 3-8 秒 —— 提前 3 秒等于一半画面是上一个球的尾巴
 * 和捡球。这个产品的核心价值就是「把捡球和等待去掉」，流拷贝直接把它毁掉。
 *
 * 为什么用 Media3 Transformer 而不是自己写 MediaCodec
 * ---------------------------------------------------
 * 精确切割意味着重编码，重编码意味着 decode→Surface→OpenGL→encode 这一整条，
 * 外加各家芯片的兼容性坑。Transformer 是 Google 官方维护的同一件事，
 * 带硬件加速，还处理了设备差异。自己写除了多一千行没有别的收益。
 *
 * 和桌面版的差别（有意的，不是偷工）
 * ---------------------------------
 *  * **末段淡出**：桌面版用 ffmpeg 的 fade/afade。Transformer 要做同样的事
 *    得写自定义的 GL 着色器和 AudioProcessor —— 留到成片能跑通之后再加，
 *    先把「切得准、接得上」这条主链验证掉。
 *  * **慢动作**：同上，而且它本来就依赖 last_hit，属于锦上添花。
 *  * **画布统一**：这个做了。多段素材有横有竖时不统一画布，
 *    输出会是变分辨率的流，很多播放器直接放不了。
 */
@UnstableApi
object Cutter {

    /** 一个待剪片段。时间单位是秒，和 core 里的 Rally 一致。 */
    data class Segment(val uri: Uri, val startS: Double, val endS: Double)

    data class Canvas(val width: Int, val height: Int)

    sealed interface Outcome {
        data class Ok(val file: File, val durationMs: Long) : Outcome
        /**
         * @param detail 失败现场，见 [describe]。**这是给人读的，不给代码判断。**
         *   Media3 的 message 常常只有「Muxer error」三个字，真正有用的东西
         *   全在 errorCode、cause 链和 ExportResult 里。
         */
        data class Failed(
            val message: String, val cause: Throwable?, val detail: String = "",
        ) : Outcome
    }

    /**
     * 把一次失败的现场摊开成人能读的几行。
     *
     * **起因是一台真机报「Muxer error」，而我们除此之外一无所知。**
     * 当时的链路是：Media3 抛 ExportException（message 就是那三个字）→
     * Cutter 只取 message → Pipeline 再 `throw RuntimeException(message)`，
     * **把 cause 整个扔掉** → 诊断报告里只剩一个指向我们自己代码的栈。
     * 一整条链下来，唯一有信息量的东西一个都没留下。
     *
     * ExportResult 里恰好有这一整轮反复在猜的答案：**真正被选中的编码器是哪个、
     * 实际用的分辨率是多少、走到第几帧才死、色彩信息是什么**。
     * 这些不该靠猜，它们一直就在参数里。
     */
    private fun describe(r: ExportResult, e: ExportException): String = buildString {
        append("错误码 ${e.getErrorCodeName()}(${e.errorCode})")
        r.videoEncoderName?.let { append("；视频编码器 $it") }
        r.audioEncoderName?.let { append("；音频编码器 $it") }
        if (r.width > 0 || r.height > 0) append("；实际输出 ${r.width}x${r.height}")
        // 走到第几帧才死：0 帧说明连第一段都没开始，几千帧说明是中途某一段
        if (r.videoFrameCount > 0) append("；已编码 ${r.videoFrameCount} 帧")
        if (r.durationMs > 0) append("；已成片 ${r.durationMs}ms")
        r.colorInfo?.let { append("；色彩 $it") }
        // cause 链才是真正的死因。Muxer 的底层异常就藏在这里。
        var c: Throwable? = e.cause
        var depth = 0
        while (c != null && depth < 4) {
            append("\n  ← ${c.javaClass.name}: ${c.message}")
            c.stackTrace.firstOrNull()?.let { append("\n      at $it") }
            c = c.cause; depth++
        }
    }

    /**
     * 选一个能装下所有素材的画布。
     *
     * 取最大宽和最大高而不是「第一段的尺寸」：竖屏 1080x1920 和横屏 1920x1080
     * 混剪时，按第一段走会把另一种裁掉一半。取包围盒再等比缩放加黑边，
     * 两种都完整保留。
     *
     * 宽高都收到偶数 —— H.264 的色度平面是 4:2:0，奇数尺寸在不少编码器上
     * 会直接失败或者悄悄裁掉一行。
     */
    /**
     * 输出画布。**对齐到 16，并且限制长边不超过 1920。**
     *
     * 原来只保证偶数（`w - w % 2`）。两个问题，都是在一台华为机器卡在
     * 「拼接成片」之后查出来的：
     *
     * **一、厂商编码器常要求 16 的倍数**，不满足时表现是**挂起而不是报错**。
     * 手机最常见的 1080p 竖屏，宽度 1080 % 16 = 8 —— 不对齐。
     * 而我们测过的素材（480x848、1024x512、544x960）**恰好全部 16 对齐**，
     * 所以这条路径一次都没走到过。
     *
     * **二、原来把源尺寸原样透传，没有任何上限。** 4K 素材就会要求编码器
     * 输出 3840x2160，而很多手机编码器根本编不了 4K，或者需要显式配 level。
     * 限制长边 1920：成片是用来看的，不是用来做母版的，而 1080p 已经
     * 超过绝大多数人分享的需要。
     *
     * 往下取整而不是往上：往上可能超过编码器的上限，往下最多损失几个像素。
     */
    fun pickCanvas(sizes: List<Canvas>): Canvas {
        if (sizes.isEmpty()) return Canvas(1280, 720)
        var w = sizes.maxOf { it.width }
        var h = sizes.maxOf { it.height }
        val long = maxOf(w, h)
        if (long > MAX_LONG_EDGE) {
            val k = MAX_LONG_EDGE.toDouble() / long
            w = (w * k).toInt()
            h = (h * k).toInt()
        }
        return Canvas(align16(w), align16(h))
    }

    /** 往下取到 16 的倍数，但不小于 16。 */
    private fun align16(v: Int): Int = maxOf(16, v - (v % 16))

    private const val MAX_LONG_EDGE = 1920

    /**
     * 剪辑并输出到 [out]。**同步**执行，调用方放到任意后台线程即可 ——
     * 内部自己开 Looper 线程给 Transformer 用，调用方不需要有 Looper。
     *
     * [onProgress] 传 0-100；Transformer 的进度是尽力而为的，
     * 拿不到时不会回调，所以调用方不能依赖它一定走到 100。
     */
    /**
     * 超时预算。**不能用一个固定的大数。**
     *
     * 原来写死 30 分钟 —— 45 秒的素材正常几秒切完，出问题时让用户盯着
     * 一个不动的进度条等满半小时，没有道理。
     *
     * 每秒成片给 20 秒预算，保底 90 秒，封顶 10 分钟。
     *
     * 20 这个倍数**实测富余很多**：Sony E5803（2015 年、骁龙 810、
     * Android 7.1）上切 22 秒成片只用 8 秒，而且走的是硬件编码
     * （OMX.qcom.video.encoder.avc），不是我原先猜的软编。
     * 留这么大余量是因为只有这一台老机器的数据，不知道更差的设备什么样；
     * 封顶 10 分钟则是因为超过这个数多半不是「慢」而是「卡住了」。
     */
    private fun deadlineFor(segments: List<Segment>): Long {
        val outS = segments.sumOf { (it.endS - it.startS).coerceAtLeast(0.0) }
        // **段数也要算进去。** 原来只按成片时长，而每一段都有固定开销：
        // 定位到非关键帧、重编码起始那个 GOP、切换输入。24 分钟素材实测
        // 出 50 段，这部分开销加起来不比编码本身少。
        //
        // 封顶从 10 分钟提到 20 分钟：50 段的活儿在中低端机上有可能超过
        // 10 分钟，那时候超时等于**在快做完的时候把成果扔掉** ——
        // 用户看到的是「等了十分钟，然后失败」，比等二十分钟拿到成片更糟。
        // 处理期间屏幕常亮、进度条一直在动，等待本身是可见的。
        val perSegment = segments.size * 5_000L
        return (outS * 20_000 + perSegment).toLong().coerceIn(90_000L, 20 * 60_000L)
    }

    fun cut(
        context: Context,
        segments: List<Segment>,
        out: File,
        canvas: Canvas,
        timeoutMs: Long = -1,          // 负数 = 按成片长度自动算，见 deadlineFor
        onProgress: ((Int) -> Unit)? = null,
        /**
         * 返回 true 表示调用方要求停下。
         *
         * **必须由调用方提供，Cutter 自己看不到协程状态。** 等待用的是
         * `lock.wait(250)`，那是阻塞等待，不响应协程取消 —— 用户点了
         * 「放弃」，job.cancel() 只标记协程取消，而这里会**继续等到超时或
         * 完成**（最长 10 分钟），Transformer 照样在编码、照样耗电，
         * HandlerThread 也一直活着。「放弃」于是变成一句空话。
         */
        shouldStop: (() -> Boolean)? = null,
    ): Outcome {
        if (segments.isEmpty()) return Outcome.Failed("没有要剪的片段", null)
        val budget = if (timeoutMs > 0) timeoutMs else deadlineFor(segments)

        val items = segments.map { seg ->
            val startMs = (seg.startS * 1000).roundToLong().coerceAtLeast(0)
            val endMs = (seg.endS * 1000).roundToLong()
            val clip = MediaItem.ClippingConfiguration.Builder()
                .setStartPositionMs(startMs)
                .setEndPositionMs(endMs)
                // false = 精确到帧（会重编码起始的那个 GOP）。
                // 设成 true 就退化成关键帧吸附，正是我们要避免的。
                .setStartsAtKeyFrame(false)
                .build()
            EditedMediaItem.Builder(
                MediaItem.Builder().setUri(seg.uri).setClippingConfiguration(clip).build()
            ).setEffects(
                Effects(
                    listOf(),
                    // SCALE_TO_FIT = 等比缩放后加黑边，不裁切。
                    // 混合方向的素材必须统一到同一画布，否则输出是变分辨率的流。
                    // 显式声明成 List<Effect>：Presentation 的具体类型推不上去，
                    // Kotlin 会拒绝 ImmutableList<Presentation> -> List<Effect>。
                    listOf<androidx.media3.common.Effect>(
                        Presentation.createForWidthAndHeight(
                            canvas.width, canvas.height, Presentation.LAYOUT_SCALE_TO_FIT
                        )
                    )
                )
            ).build()
        }

        val seq = ImmutableList.of(EditedMediaItemSequence.Builder(items).build())

        // **HDR 素材要单独处理，而且没有一种模式在所有设备上都行。**
        //
        // 不设 hdrMode 时 Media3 默认 HDR_MODE_KEEP_HDR（原样保留 HDR），
        // 而输出被写死成 H.264 —— H.264 配 HLG/PQ 几乎没有设备支持。
        // 实测（HdrSourceTest，一段 BT.2020 + HLG + 10bit 的合成素材）：
        // 默认模式直接 Video frame processing error。
        //
        // 但换成色调映射也不是稳赢：同一个测试在模拟器上会卡在**解码器** ——
        // goldfish 解码器接不了 KEY_COLOR_TRANSFER_REQUEST，而两种色调映射
        // 模式都要设它。也就是说「压成 SDR」在有些设备上反而更糟。
        //
        // 所以按顺序试，第一个成功的算数：
        //  1. 色调映射到 SDR（我们的输出本来就是 SDR H.264，这是语义上对的）
        //  2. 退回默认 —— 万一这台机器真能编 HDR H.264
        // 只在**确实是 HDR 素材**时才多试一次；SDR 素材（到今天为止所有
        // 跑通过的素材）一次直接走完，路径和以前完全一样，不引入新风险。
        val hdr = segments.any { isHdr(context, it.uri) }
        val modes = if (!hdr) listOf<Int?>(null) else listOf(
            Composition.HDR_MODE_TONE_MAP_HDR_TO_SDR_USING_OPEN_GL,
            null,
        )

        var last: Outcome = Outcome.Failed("没有可用的导出方式", null)
        for ((i, mode) in modes.withIndex()) {
            val composition = Composition.Builder(seq)
                .apply { if (mode != null) setHdrMode(mode) }
                .build()
            last = export(context, composition, out, budget, onProgress, shouldStop)
            // 成功、被用户放弃、或者已经是最后一次 —— 都不再试。
            // **超时不重试**：预算已经烧掉一次了，再来一次是让用户等两倍。
            if (last is Outcome.Ok || last is Outcome.Failed && last.message == "已取消") break
            if (i == modes.lastIndex) break
            if (last is Outcome.Failed && last.message.startsWith("超时")) break
        }
        return last
    }

    /**
     * 这段素材是不是 HDR。
     *
     * 只看颜色传输特性：PQ(ST2084) 和 HLG 是两种 HDR 曲线，其余都当 SDR。
     * 读不出来就当 SDR —— **判错方向要选安全的那边**：把 HDR 当 SDR 只是
     * 少试一次色调映射（还有第二轮兜底），把 SDR 当 HDR 则会给所有正常
     * 素材加一条没验证过的路径。
     */
    internal fun isHdr(context: Context, uri: Uri): Boolean = runCatching {
        val ex = android.media.MediaExtractor()
        try {
            ex.setDataSource(context, uri, null)
            for (i in 0 until ex.trackCount) {
                val f = ex.getTrackFormat(i)
                val mime = f.getString(android.media.MediaFormat.KEY_MIME) ?: continue
                if (!mime.startsWith("video/")) continue
                if (!f.containsKey(android.media.MediaFormat.KEY_COLOR_TRANSFER)) continue
                val t = f.getInteger(android.media.MediaFormat.KEY_COLOR_TRANSFER)
                if (t == android.media.MediaFormat.COLOR_TRANSFER_ST2084 ||
                    t == android.media.MediaFormat.COLOR_TRANSFER_HLG) return true
            }
        } finally { ex.release() }
        false
    }.getOrDefault(false)

    /** 跑一次导出，同步等结果。[cut] 可能会用不同的 HDR 模式调它两次。 */
    private fun export(
        context: Context,
        composition: Composition,
        out: File,
        budget: Long,
        onProgress: ((Int) -> Unit)?,
        shouldStop: (() -> Boolean)?,
    ): Outcome {
        // **Transformer 把「构造它的那条线程」当成自己的应用线程**，之后
        // start / getProgress / cancel 全都必须回到同一条线程，否则直接
        // IllegalStateException: accessed on the wrong thread。
        //
        // 而且它是在那条线程的 Looper 上跑消息循环的 —— 所以也绝不能在
        // 同一条线程上 start() 之后阻塞等结果，Looper 被占住就永远不会完成。
        // 两个坑都踩过：先是全在调用方线程上做（死锁，表现为「不转码的测试
        // 通过、真转码的全超时」，看起来像模拟器慢），改成分开之后又撞上
        // 线程校验。正确做法是 Cutter 自己开一条 Looper 线程，
        // **构造和所有调用都放进去**，阻塞的是调用方。
        val worker = HandlerThread("pipo-cut").apply { start() }
        val handler = Handler(worker.looper)
        var result: Outcome? = null
        var transformer: Transformer? = null
        val lock = Object()

        handler.post {
            val t = Transformer.Builder(context)
                .setVideoMimeType(MimeTypes.VIDEO_H264)
                .setAudioMimeType(MimeTypes.AUDIO_AAC)
                .addListener(object : Transformer.Listener {
                    override fun onCompleted(c: Composition, r: ExportResult) {
                        synchronized(lock) {
                            result = Outcome.Ok(out, r.durationMs); lock.notifyAll()
                        }
                    }

                    override fun onError(c: Composition, r: ExportResult, e: ExportException) {
                        synchronized(lock) {
                            result = Outcome.Failed(e.message ?: "导出失败", e, describe(r, e))
                            lock.notifyAll()
                        }
                    }
                })
                .build()
            synchronized(lock) { transformer = t }
            t.start(composition, out.absolutePath)
        }

        val deadline = System.currentTimeMillis() + budget
        val holder = ProgressHolder()
        var stopped = false
        synchronized(lock) {
            while (result == null && System.currentTimeMillis() < deadline) {
                lock.wait(250)
                if (shouldStop?.invoke() == true) { stopped = true; break }
                if (onProgress != null) {
                    val t = transformer
                    if (t != null) handler.post {
                        if (t.getProgress(holder) == Transformer.PROGRESS_STATE_AVAILABLE) {
                            onProgress(holder.progress)
                        }
                    }
                }
            }
        }
        val r = result
        if (r == null) {
            // 超时或被放弃都要停掉，否则编码器一直被占着
            val t = transformer
            if (t != null) handler.post { runCatching { t.cancel() } }
            // 半截的成片留着没意义，而且会被当成「上次的输出」占空间
            runCatching { if (out.exists()) out.delete() }
        }
        handler.post { worker.quitSafely() }
        return r ?: if (stopped) Outcome.Failed("已取消", null)
                    else Outcome.Failed("超时：超过 ${budget / 1000} 秒仍未完成", null)
    }
}
