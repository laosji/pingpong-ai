package cc.pipo.cutter

import android.net.Uri
import androidx.media3.common.util.UnstableApi
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * 剪辑失败时，我们到底留下了什么。
 *
 * 起因
 * ----
 * 一台真机上报「Muxer error」，除此之外**一无所知**。查下来是我们自己
 * 把信息丢光了：Media3 抛的 ExportException，message 就只有那三个字，
 * 有用的东西在 errorCode、cause 链和 ExportResult 里；而 Cutter 只取
 * message，Pipeline 再 `throw RuntimeException(message)` 把 cause 也扔了。
 *
 * 所以这个测试不测「剪辑成功」，它测的是**失败的时候我们能不能说清楚**。
 * 做法是要一个这台设备的编码器一定接不了的画布（远超能力表上限），
 * 让导出必然失败，然后检查现场里有没有真东西。
 *
 * 这类断言容易写成永远通过（非空、包含冒号之类），所以下面逐项查的是
 * **具体的字段**：错误码名字、真正被选中的编码器、实际分辨率。
 */
@RunWith(AndroidJUnit4::class)
@UnstableApi
class FailureDetailTest {

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
     * 怎么可靠地造一次失败。
     *
     * **第一版试的是「要一个超过编码器上限的画布」，结果它成功了** ——
     * Media3 的 DefaultEncoderFactory 自带分辨率回退，会自己降到能编的尺寸。
     * 顺带说明：这一轮怀疑过的「画布尺寸不被编码器接受」，很可能从一开始
     * 就不成立，因为 Media3 本来就会调。
     *
     * 改成写到不可写的路径：muxer 建不出来，必然失败，而且**打中的正是
     * 真机上报的那一类**（muxer / IO），不挑设备。
     */
    @Test
    fun 导出失败时现场里要有错误码和编码器() {
        val src = Uri.fromFile(asset("land.mp4"))
        // /system 在任何未 root 的设备上都不可写
        val out = File("/system/pipo_should_not_be_writable.mp4")
        println("  故意写到不可写路径 ${out.absolutePath}")

        val res = Cutter.cut(
            ctx, listOf(Cutter.Segment(src, 1.0, 3.0)), out, Cutter.Canvas(320, 180),
            timeoutMs = 90_000)

        assertTrue("本该失败却成功了 —— /system 居然可写？换个路径再试",
            res is Cutter.Outcome.Failed)
        val f = res as Cutter.Outcome.Failed
        println("  message = ${f.message}")
        println("  detail  = ${f.detail}")

        // **cause 必须留着。** 之前整条链都在丢它。
        assertTrue("cause 丢了 —— 真正的死因就在这里面", f.cause != null)

        // 错误码名字：Media3 的 message 可能只有「Muxer error」，
        // 错误码是唯一稳定可分类的东西
        assertTrue("现场里没有错误码：${f.detail}",
            f.detail.contains("ERROR_CODE_"))

        // **底层死因必须出现在现场里。** 这才是「Muxer error」之外真正
        // 有信息量的那一段 —— 之前整条链都把它丢了。
        assertTrue("现场里没有 cause 链：${f.detail}", f.detail.contains("←"))
    }
}
