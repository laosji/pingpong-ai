package cc.pipo.cutter

import android.media.MediaCodecList
import android.net.Uri
import androidx.media3.common.util.UnstableApi
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * 很多段拼一支片，在**真硬件编码器**上。
 *
 * 为什么是这个组合
 * ----------------
 * 一台真机剪 23:16 的素材报「Muxer error」。同一个文件在模拟器上**成功了**
 * ——117 段、9:22、94.4MB。所以不是素材的问题，是设备的问题；而模拟器和
 * 手机之间唯一的实质差别就是**软件编码器 vs 厂商硬件编码器**
 * （模拟器只有 c2.android/OMX.google，手机上是 OMX.qcom / OMX.hisi）。
 *
 * 段数也是这次才第一次撞到的量级：以前测过的最多 50 段，这次是 117。
 * 每一段都要定位到非关键帧、重编码起始 GOP、再喂给同一个 muxer。
 *
 * 所以这个测试把两个变量凑齐：段数拉到 120，且**只在有硬件编码器的机器上
 * 才真正有意义**（没有的机器照跑，只是证明不了什么，会在输出里说明）。
 *
 * 素材用手上的 land.mp4（合成的），不涉及任何人的录像。
 */
@RunWith(AndroidJUnit4::class)
@UnstableApi
class ManySegmentsTest {

    /** **必须走 logcat，不能用 println。**
     * Test Lab 上 println 不落 logcat —— 而机型矩阵正是这些测试
     * 唯一有价值的地方（本地只有一台模拟器 + 一台 Sony）。
     * 结果只剩「过/不过」，各家编码器的实际参数一个都拿不到。 */
    private val LOG = "PipoManySegments"

    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext

    private fun asset(name: String): File {
        val out = File(ctx.cacheDir, name)
        if (!out.exists() || out.length() == 0L) {
            InstrumentationRegistry.getInstrumentation().context.assets.open(name).use { i ->
                out.outputStream().use { o -> i.copyTo(o) }
            }
        }
        return out
    }

    /** 这台机器上跑的是不是厂商硬件编码器。名字最可靠：软件的一律 google/c2.android。 */
    private fun hardwareEncoder(): String? =
        MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos
            .filter { it.isEncoder && it.supportedTypes.any { t -> t.equals("video/avc", true) } }
            .map { it.name }
            .firstOrNull { n ->
                !n.startsWith("OMX.google", true) && !n.startsWith("c2.android", true)
            }

    @Test
    fun 一百二十段能拼成一支片() {
        val f = asset("land.mp4")
        val src = Uri.fromFile(f)
        val hw = hardwareEncoder()
        android.util.Log.i(LOG, "  硬件编码器 ${hw ?: "没有（这台只有软编，测不到厂商实现）"}")

        // land.mp4 有十几秒。在里面反复取 0.4 秒的小段，凑到 120 段 ——
        // **每段都不落在关键帧上**，逼 Transformer 走重编码那条路，
        // 和真实成片一样。
        val segs = (0 until 120).map { i ->
            val start = 1.0 + (i % 28) * 0.43
            Cutter.Segment(src, start, start + 0.4)
        }
        val out = File(ctx.cacheDir, "many120.mp4").also { it.delete() }
        val canvas = Cutter.pickCanvas(listOf(Cutter.Canvas(320, 180)))

        val t0 = System.currentTimeMillis()
        val res = Cutter.cut(ctx, segs, out, canvas, timeoutMs = 20 * 60 * 1000)
        val took = System.currentTimeMillis() - t0
        android.util.Log.i(LOG, "  ${segs.size} 段用了 ${took / 1000}s -> $res")
        android.util.Log.i(LOG, "  输出 ${out.length()} 字节")

        if (res is Cutter.Outcome.Failed) {
            // 现场是这个测试真正的产出：**失败比成功有价值**，
            // 前提是失败时说得清楚。
            android.util.Log.i(LOG, "  失败现场：\n${res.detail}")
        }
        assertTrue("120 段拼片失败" + (if (hw != null) "（硬件编码器 $hw）" else "") +
            "：${(res as? Cutter.Outcome.Failed)?.let { it.message + "\n" + it.detail } ?: res}",
            res is Cutter.Outcome.Ok)
        assertTrue("输出是空的", out.length() > 0)
    }
}
