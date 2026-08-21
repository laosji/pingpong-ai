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
        data class Failed(val message: String, val cause: Throwable?) : Outcome
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
    fun pickCanvas(sizes: List<Canvas>): Canvas {
        if (sizes.isEmpty()) return Canvas(1280, 720)
        val w = sizes.maxOf { it.width }
        val h = sizes.maxOf { it.height }
        return Canvas(w - (w % 2), h - (h % 2))
    }

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
        return (outS * 20_000).toLong().coerceIn(90_000L, 10 * 60_000L)
    }

    fun cut(
        context: Context,
        segments: List<Segment>,
        out: File,
        canvas: Canvas,
        timeoutMs: Long = -1,          // 负数 = 按成片长度自动算，见 deadlineFor
        onProgress: ((Int) -> Unit)? = null,
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

        val composition = Composition.Builder(
            ImmutableList.of(EditedMediaItemSequence.Builder(items).build())
        ).build()

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
                            result = Outcome.Failed(e.message ?: "导出失败", e); lock.notifyAll()
                        }
                    }
                })
                .build()
            synchronized(lock) { transformer = t }
            t.start(composition, out.absolutePath)
        }

        val deadline = System.currentTimeMillis() + budget
        val holder = ProgressHolder()
        synchronized(lock) {
            while (result == null && System.currentTimeMillis() < deadline) {
                lock.wait(250)
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
            // 超时也要停掉，否则那条线程会继续占着编码器
            val t = transformer
            if (t != null) handler.post { runCatching { t.cancel() } }
        }
        handler.post { worker.quitSafely() }
        return r ?: Outcome.Failed("超时：超过 ${budget / 1000} 秒仍未完成", null)
    }
}
