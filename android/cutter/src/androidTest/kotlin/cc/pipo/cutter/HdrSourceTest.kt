package cc.pipo.cutter

import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import androidx.media3.common.util.UnstableApi
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * HDR 素材能不能剪。
 *
 * 为什么单独拎出来
 * ----------------
 * 华为那台机器卡在「拼接成片」，三个嫌疑里前两个（画布不对齐、没有分辨率
 * 上限）已经改掉了，**这是剩下的第三个**：现在的手机（华为尤其）默认录
 * 10-bit HLG，而我们的输出被写死成 H.264。
 *
 * 素材是 ffmpeg 合成的（testsrc2 + 正弦音），不是任何人的录像 ——
 * 色彩描述照着手机录像写：BT.2020 / HLG(arib-std-b67) / yuv420p10le，
 * 1080x1920 顺带把「宽度 1080 不是 16 的倍数」也一起覆盖了。
 *
 * 关于 assume
 * -----------
 * 剪辑那条在模拟器上会被跳过，因为 goldfish 解码器**接不了色调映射请求**
 * （见下面的探测）。这是**跑之前先探**，不是失败之后看错误消息找借口 ——
 * 后者等于给自己开脱，任何解码器故障都能被说成「环境问题」。
 * 前两条（识别 HDR、画布对齐）在哪里都真跑。
 */
@RunWith(AndroidJUnit4::class)
@UnstableApi
class HdrSourceTest {

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

    /**
     * 这台设备能不能一边解 10-bit HEVC、一边按要求把输出转成 SDR。
     *
     * 两种色调映射模式都要给解码器设 KEY_COLOR_TRANSFER_REQUEST，
     * 所以这就是那条路径能不能走的先决条件。直接 configure 一次问 ——
     * 比查能力表准，因为能力表里没有这一项。
     */
    private fun canToneMap(src: File): Boolean {
        val ex = MediaExtractor()
        return try {
            ex.setDataSource(src.absolutePath)
            var fmt: MediaFormat? = null
            for (i in 0 until ex.trackCount) {
                val f = ex.getTrackFormat(i)
                if (f.getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true) {
                    fmt = f; break
                }
            }
            if (fmt == null) return false
            fmt.setInteger(MediaFormat.KEY_COLOR_TRANSFER_REQUEST,
                MediaFormat.COLOR_TRANSFER_SDR_VIDEO)
            val name = android.media.MediaCodecList(android.media.MediaCodecList.REGULAR_CODECS)
                .findDecoderForFormat(fmt) ?: return false
            val codec = MediaCodec.createByCodecName(name)
            try {
                // 不给 Surface：只问「这个格式你收不收」，不真解
                codec.configure(fmt, null, null, 0)
                true
            } catch (e: Exception) {
                println("  这台设备接不了色调映射请求：$e")
                false
            } finally {
                runCatching { codec.release() }
            }
        } catch (e: Exception) {
            println("  探测失败：$e"); false
        } finally {
            runCatching { ex.release() }
        }
    }

    @Test
    fun 能认出这是HDR素材() {
        val src = Uri.fromFile(asset("hdr.mp4"))
        assertTrue("没认出 HLG 素材是 HDR —— 那么色调映射永远不会启用",
            Cutter.isHdr(ctx, src))
        // 反面：手上的 SDR 素材不能被误判，否则所有正常素材都被推上新路径
        assertTrue("把 SDR 素材误判成 HDR", !Cutter.isHdr(ctx, Uri.fromFile(asset("land.mp4"))))
    }

    @Test
    fun 竖屏1080的画布是16的倍数() {
        val c = Cutter.pickCanvas(listOf(Cutter.Canvas(1080, 1920)))
        println("  1080x1920 -> ${c.width}x${c.height}")
        assertTrue("宽 ${c.width} 不是 16 的倍数", c.width % 16 == 0)
        assertTrue("高 ${c.height} 不是 16 的倍数", c.height % 16 == 0)
    }

    @Test
    fun HLG十比特素材能剪出成片() {
        val f = asset("hdr.mp4")
        assumeTrue("这台设备的解码器接不了色调映射请求，跳过（不是我们的 bug）",
            canToneMap(f))

        val out = File(ctx.cacheDir, "hdr_out.mp4").also { it.delete() }
        val canvas = Cutter.pickCanvas(listOf(Cutter.Canvas(1080, 1920)))
        val res = Cutter.cut(
            ctx, listOf(Cutter.Segment(Uri.fromFile(f), 0.5, 3.0)), out, canvas,
            timeoutMs = 3 * 60 * 1000)

        println("  结果 $res")
        // 超时和报错要分开看：**超时才是用户看到的「卡住」**，
        // 报错至少还会弹个提示。两种都不接受，但区分开有助于定位。
        val timedOut = res is Cutter.Outcome.Failed && res.message.startsWith("超时")
        assertTrue("HDR 素材超时了 —— 这就是用户那边的「卡在拼接成片」", !timedOut)
        assertTrue("HDR 素材剪失败：$res", res is Cutter.Outcome.Ok)
        assertTrue("输出文件是空的", out.length() > 0)
    }
}
