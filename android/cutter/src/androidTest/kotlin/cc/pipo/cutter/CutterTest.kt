package cc.pipo.cutter

import android.media.MediaMetadataRetriever
import android.net.Uri
import androidx.media3.common.util.UnstableApi
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.abs

/**
 * 在真机/模拟器上验证切片。
 *
 * 这个测试要回答的是「技术上到底行不行」，不是「代码能不能编译」：
 *
 *  1. **切点准不准** —— 这是选 Transformer 而不是 MediaExtractor 的全部理由。
 *     实测手机录像关键帧间隔 1-3 秒，流拷贝会让片段提前那么多开始；
 *     这里要证明重编码路径能把误差压到帧级。
 *  2. **混合方向能不能接起来** —— 一横一竖两段素材，输出必须是单一分辨率，
 *     否则很多播放器直接放不了。
 *  3. **多段能不能接** —— 成片本来就是十几段拼的。
 */
@RunWith(AndroidJUnit4::class)
@UnstableApi
class CutterTest {

    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext

    /** 把 androidTest 的 asset 拷到可读路径 —— Transformer 要的是真实文件。 */
    private fun asset(name: String): File {
        val out = File(ctx.cacheDir, name)
        if (!out.exists() || out.length() == 0L) {
            InstrumentationRegistry.getInstrumentation().context.assets.open(name).use { i ->
                out.outputStream().use { o -> i.copyTo(o) }
            }
        }
        return out
    }

    private fun durationMs(f: File): Long {
        val r = MediaMetadataRetriever()
        return try {
            r.setDataSource(f.absolutePath)
            r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)!!.toLong()
        } finally { r.release() }
    }

    private fun size(f: File): Pair<Int, Int> {
        val r = MediaMetadataRetriever()
        return try {
            r.setDataSource(f.absolutePath)
            val w = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)!!.toInt()
            val h = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)!!.toInt()
            // 旋转元数据会让「宽高」和实际显示方向不一致，这里按显示方向归一
            val rot = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)
                ?.toIntOrNull() ?: 0
            if (rot == 90 || rot == 270) h to w else w to h
        } finally { r.release() }
    }

    /**
     * Cutter 内部自己开 Looper 线程，所以这里直接调即可。
     * **不要**把它 post 到某条 Looper 线程再阻塞等 —— 那正是第一版的死锁写法。
     */
    private fun runCut(
        segments: List<Cutter.Segment>, out: File, canvas: Cutter.Canvas,
    ): Cutter.Outcome = Cutter.cut(ctx, segments, out, canvas, timeoutMs = 6 * 60 * 1000)

    @Test
    fun 单段切割的时长误差在一帧以内() {
        val src = Uri.fromFile(asset("land.mp4"))
        val out = File(ctx.cacheDir, "one.mp4").also { it.delete() }
        // 故意选一个**不在关键帧上**的起点：关键帧每 1 秒一个，
        // 4.37 秒离最近的关键帧有 370 毫秒 —— 流拷贝会在这里露馅。
        val res = runCut(
            listOf(Cutter.Segment(src, 4.37, 7.12)), out, Cutter.Canvas(320, 180))
        assertTrue("剪辑失败：$res", res is Cutter.Outcome.Ok)
        assertTrue("输出文件为空", out.length() > 0)
        val want = ((7.12 - 4.37) * 1000).toLong()
        val got = durationMs(out)
        // 一帧 = 33 毫秒（30fps）。给 2 帧余量：编码器可能会补/丢一帧。
        assertTrue("时长 ${got}ms，期望约 ${want}ms（差 ${abs(got - want)}ms）",
            abs(got - want) <= 70)
        println("  单段：期望 ${want}ms，实际 ${got}ms，差 ${abs(got - want)}ms")
    }

    @Test
    fun 多段拼接时长相加() {
        val src = Uri.fromFile(asset("land.mp4"))
        val out = File(ctx.cacheDir, "many.mp4").also { it.delete() }
        val segs = listOf(
            Cutter.Segment(src, 2.5, 4.0),
            Cutter.Segment(src, 7.3, 8.8),
            Cutter.Segment(src, 11.0, 12.4),
        )
        val res = runCut(segs, out, Cutter.Canvas(320, 180))
        assertTrue("剪辑失败：$res", res is Cutter.Outcome.Ok)
        val want = segs.sumOf { (it.endS - it.startS) * 1000 }.toLong()
        val got = durationMs(out)
        assertTrue("时长 ${got}ms，期望约 ${want}ms", abs(got - want) <= 200)
        println("  三段：期望 ${want}ms，实际 ${got}ms")
    }

    @Test
    fun 横竖混剪输出单一分辨率() {
        val land = Uri.fromFile(asset("land.mp4"))   // 320x180
        val port = Uri.fromFile(asset("port.mp4"))   // 136x240
        val out = File(ctx.cacheDir, "mixed.mp4").also { it.delete() }
        val canvas = Cutter.pickCanvas(
            listOf(Cutter.Canvas(320, 180), Cutter.Canvas(136, 240)))
        assertEquals("画布应取包围盒", Cutter.Canvas(320, 240), canvas)

        val res = runCut(
            listOf(Cutter.Segment(land, 1.0, 3.0), Cutter.Segment(port, 2.0, 4.0)),
            out, canvas)
        assertTrue("剪辑失败：$res", res is Cutter.Outcome.Ok)
        val (w, h) = size(out)
        assertEquals("输出宽", canvas.width, w)
        assertEquals("输出高", canvas.height, h)
        println("  混剪：输出 ${w}x${h}，时长 ${durationMs(out)}ms")
    }

    @Test
    fun 画布取包围盒且收成偶数() {
        assertEquals(Cutter.Canvas(1920, 1920),
            Cutter.pickCanvas(listOf(Cutter.Canvas(1920, 1080), Cutter.Canvas(1080, 1920))))
        // 奇数会被收成偶数：H.264 的 4:2:0 色度平面吃不下奇数尺寸
        assertEquals(Cutter.Canvas(640, 358),
            Cutter.pickCanvas(listOf(Cutter.Canvas(641, 359))))
    }
}
