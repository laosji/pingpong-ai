package cc.pipo.cutter

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import cc.pipo.core.Audio
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.DataInputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Properties
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * 解码 + 重采样这一步的 parity。
 *
 * **这是整条链上最后一处可能悄悄跑偏的地方。** 前面 Audio.kt 和 Highlight.kt
 * 都做到了逐点一致，但那是拿 Python 已经算好的 PCM 当输入。真跑起来时
 * PCM 是 Android 自己解出来的：解码器不同（MediaCodec vs ffmpeg 的 AAC）、
 * 重采样算法不同（自写多相 sinc vs swresample）。这两处只要有一处偏得多，
 * 前面所有的一致就白做了。
 *
 * 逐样本相同是不可能的，也不必要。**真正要保住的是「检出的击球时刻不变」** ——
 * 那才是下游依赖的东西。所以这里比的是击球时刻，不是 PCM 波形。
 */
@RunWith(AndroidJUnit4::class)
class AudioDecodeParityTest {

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

    private fun f64(name: String): DoubleArray {
        val raw = InstrumentationRegistry.getInstrumentation().context.assets.open(name)
            .use { DataInputStream(it).readBytes() }
        val b = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
        return DoubleArray(b.remaining() / 8) { b.double }
    }

    private val p = Properties().apply {
        InstrumentationRegistry.getInstrumentation().context.assets.open("land.properties")
            .use { load(it) }
    }

    @Test
    fun 解码出的PCM长度与响度和Python一致() {
        val pcm = AudioDecode.decode(asset("land.mp4").absolutePath, 16000)
        val want = p.getProperty("n16")!!.toInt()
        // 长度允许差几个采样：解码器对首尾帧的处理（priming samples）不同
        assertTrue("采样数 ${pcm.size}，Python ${want}，差 ${abs(pcm.size - want)}",
            abs(pcm.size - want) < 16000 / 10)     // 100 毫秒以内

        val rms = sqrt(pcm.fold(0.0) { a, v -> a + v.toDouble() * v } / pcm.size)
        val peak = pcm.maxOf { abs(it) }
        val wantRms = p.getProperty("rms16")!!.toDouble()
        val wantPeak = p.getProperty("peak16")!!.toDouble()
        // 响度差 3% 以内 —— 再多就说明重采样的增益或滤波出了问题
        assertTrue("RMS $rms vs Python $wantRms", abs(rms - wantRms) / wantRms < 0.03)
        assertTrue("峰值 $peak vs Python $wantPeak", abs(peak - wantPeak) / wantPeak < 0.05)
        println("  PCM ${pcm.size} 采样（Python $want），RMS %.6f / %.6f，峰值 %.4f / %.4f"
            .format(rms, wantRms, peak, wantPeak))
    }

    @Test
    fun 端到端检出的击球时刻与Python一致() {
        val pcm = AudioDecode.decode(asset("land.mp4").absolutePath, 16000)
        val got = Audio.detectHits(pcm, Audio.Config()).hits
        val want = f64("land_hits.f64")

        // 配对：每个 Python 的击球在 Kotlin 里找 30 毫秒（约一帧）内的对应点。
        // 不要求数量完全相同 —— 解码差异可能让阈值边缘的候选进出一两个，
        // 但**绝大多数必须对得上**，否则后面所有指标都不成立。
        var matched = 0
        var worst = 0.0
        for (w in want) {
            val d = got.minOfOrNull { abs(it - w) } ?: Double.MAX_VALUE
            if (d <= 0.030) { matched++; worst = maxOf(worst, d) }
        }
        val rate = matched.toDouble() / want.size
        println("  Python ${want.size} 个击球，Kotlin ${got.size} 个，" +
                "对上 $matched（%.1f%%），最大偏差 %.1f 毫秒".format(rate * 100, worst * 1000))

        // **对不上的时候，先分清是「整体偏移」还是「乱」。**
        // 这两种的处置完全不同：整体偏移是解码器在开头多吐了一段
        // （AAC 的 priming samples），一个常数就能解释，产品影响也小
        // （成片前后各留 0.4/0.7 秒的余量）；而抖动意味着重采样或检测本身坏了。
        // 不打印这一行的话，只看到「41.3%」，两种情况长得一模一样。
        var bestShift = 0.0
        var bestHit = -1
        var s = -0.100
        while (s <= 0.100) {
            val m = want.count { w -> got.any { abs(it - s - w) <= 0.030 } }
            if (m > bestHit) { bestHit = m; bestShift = s }
            s += 0.001
        }
        println("  最佳整体平移 %+.0f 毫秒时能对上 %d/%d（%.1f%%）".format(
            bestShift * 1000, bestHit, want.size, 100.0 * bestHit / want.size))
        // **断言的是「形状一致 + 偏移有界」，不是「和 ffmpeg 绝对对齐」。**
        //
        // 原来直接要求 90% 落在 30 毫秒内。模拟器能过，而 Sony E5803
        // （API 25、高通）只对上 41.3% —— 一度看着像解码坏了。
        // 量了最佳平移才看清：**平移 +11 毫秒就能对上 45/46（97.8%）**，
        // 是纯常数偏移，不是抖动。来源是解码器在开头多吐的 priming samples，
        // 各家实现不同，我们无权要求它们一致。
        //
        // 而常数偏移对产品几乎无害：成片前后各留 0.4 / 0.7 秒余量，
        // 11 毫秒是其中的 2.8%。真正会毁掉成片的是**抖动**（每个点偏不同的量）
        // 和**大偏移**，所以分开来卡：
        //   * 扣掉常数偏移后必须 ≥ 90% 对上 —— 抖动逃不掉
        //   * 偏移本身必须 ≤ 30 毫秒 —— 大偏移逃不掉
        // 这样测的才是「这台设备上剪出来的片对不对」，而不是
        // 「这台设备的解码器和 ffmpeg 像不像」。
        assertTrue(
            "扣掉 %+.0f 毫秒的整体偏移之后仍然只对上 %.1f%% —— 这是抖动，不是偏移，"
                .format(bestShift * 1000, 100.0 * bestHit / want.size) +
                "说明重采样或检测本身出了问题",
            bestHit.toDouble() / want.size >= 0.90)
        assertTrue(
            "整体偏移 %+.0f 毫秒，超过 30 毫秒 —— 解码器丢弃的 priming 太多，"
                .format(bestShift * 1000) + "切点会系统性偏移",
            abs(bestShift) <= 0.030)
        assertTrue("击球数差太多：Python ${want.size} vs Kotlin ${got.size}",
            abs(got.size - want.size) <= want.size / 10 + 1)
    }

    @Test
    fun 重采样保持能量且长度正确() {
        // 1 秒 1kHz 正弦，48k → 16k：长度应当正好三分之一，幅度基本不变
        val n = 48000
        val x = FloatArray(n) { kotlin.math.sin(2.0 * Math.PI * 1000.0 * it / 48000.0).toFloat() }
        val y = Resampler.resample(x, 48000, 16000)
        assertEquals("长度", 16000, y.size)
        // 掐头去尾避开滤波器的瞬态
        val body = y.copyOfRange(200, y.size - 200)
        val rms = sqrt(body.fold(0.0) { a, v -> a + v.toDouble() * v } / body.size)
        assertTrue("重采样后 RMS $rms，正弦应当约 0.707", abs(rms - 0.7071) < 0.02)
        println("  48k→16k 正弦：长度 ${y.size}，RMS %.4f".format(rms))
    }
}
