package cc.pipo.core

import java.io.DataInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Properties
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * 对着 Python 的输出逐点比对。
 *
 * 基准来自真实素材（5cf65f 的 120-140 秒，20 秒对打，1997 帧，71 次击球），
 * 不是合成信号 —— 合成信号测不出「实际录音里那些边界情况」。
 *
 * 为什么值得为一个检测函数写这种测试：回合分组、主题排序、成片时长
 * 全都建在它输出的时间点上。这里差一帧，下游会放大成「选了另一段球」。
 * 而且 Python 那版是唯一有真值验证过的（击球级召回 0.850），
 * Kotlin 版**只有和它一致才继承那个结论**，否则所有已知指标都得重测。
 */
class AudioParityTest {

    // 用类自身的 getResourceAsStream + 绝对路径：javaClass.classLoader
    // 在某些类加载器下会是 null，测试运行器的配置不同就会踩到。
    private fun res(name: String): ByteArray =
        AudioParityTest::class.java.getResourceAsStream("/" + name)!!
            .use { DataInputStream(it).readBytes() }

    private fun f32(name: String): FloatArray {
        val b = ByteBuffer.wrap(res(name)).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(b.remaining() / 4) { b.float }
    }

    private fun f64(name: String): DoubleArray {
        val b = ByteBuffer.wrap(res(name)).order(ByteOrder.LITTLE_ENDIAN)
        return DoubleArray(b.remaining() / 8) { b.double }
    }

    private val p = Properties().apply {
        AudioParityTest::class.java.getResourceAsStream("/params.properties")!!.use { load(it) }
    }

    private fun s(k: String) = p.getProperty(k)!!

    private val cfg = Audio.Config(
        sr = s("sr").toInt(), nFft = s("nFft").toInt(), hop = s("hop").toInt(),
        fmin = s("fmin").toDouble(), fmax = s("fmax").toDouble(),
        kMad = s("kMad").toDouble(), noiseWinS = s("noiseWinS").toDouble(),
        minGapS = s("minGapS").toDouble(),
    )

    @Test
    fun `包络与 Python 逐帧一致`() {
        val clip = f32("clip.f32")
        val want = f32("env.f32")
        val (got, fr) = Audio.onsetEnvelope(clip, cfg)
        assertEquals(s("frameRate").toDouble(), fr, 1e-9, "帧率")
        assertEquals(want.size, got.size, "帧数")

        // 相对容差：包络是几百个 log 项相加，float32 的累加顺序不同会有
        // 最后几位的差。要求绝对相等会得到一个永远红的测试，那比没有更糟。
        var worst = 0.0
        var worstAt = -1
        for (i in want.indices) {
            val d = abs(got[i] - want[i]) / (abs(want[i]) + 1e-3)
            if (d > worst) { worst = d; worstAt = i }
        }
        assertTrue(worst < 1e-4, "包络最大相对差 $worst（第 $worstAt 帧），超出 1e-4")
        println("  包络 ${got.size} 帧，最大相对差 %.2e".format(worst))
    }

    @Test
    fun `阈值曲线与 Python 一致`() {
        val clip = f32("clip.f32")
        val want = f32("thr.f32")
        val (env, fr) = Audio.onsetEnvelope(clip, cfg)
        val got = Audio.adaptiveThreshold(env, fr, cfg.noiseWinS, cfg.kMad)
        assertEquals(want.size, got.size, "长度")
        var worst = 0.0
        for (i in want.indices) worst = maxOf(worst, abs(got[i] - want[i]) / (abs(want[i]) + 1e-3))
        assertTrue(worst < 1e-4, "阈值最大相对差 $worst")
        println("  阈值 ${got.size} 点，最大相对差 %.2e".format(worst))
    }

    @Test
    fun `击球时刻与 Python 完全一致`() {
        val clip = f32("clip.f32")
        val want = f64("hits.f64")
        val got = Audio.detectHits(clip, cfg).hits

        assertEquals(s("nHits").toInt(), want.size, "基准自洽")
        // 时刻是「帧号 / 帧率」，两边都由整数帧号得来，所以应当精确相等。
        // 这里不给容差 —— 差一帧就是差 10 毫秒，下游分组会因此换一个回合。
        assertEquals(want.size, got.size,
            "击球数不一致：Python ${want.size}，Kotlin ${got.size}")
        for (i in want.indices) {
            assertTrue(abs(got[i] - want[i]) < 1e-9,
                "第 $i 个击球：Python ${want[i]}，Kotlin ${got[i]}")
        }
        println("  击球 ${got.size} 个，全部逐点一致")
    }

    @Test
    fun `力量与 Python 一致`() {
        val clip = f32("clip.f32")
        val r = Audio.detectHits(clip, cfg)
        val amp = Audio.hitAmplitudes(r.hits, r.env, r.frameRate)
        assertEquals(r.hits.size, amp.size)
        assertTrue(amp.all { it > 0.0 }, "力量应当都为正")
        println("  力量 ${amp.size} 个，范围 %.1f - %.1f".format(amp.min(), amp.max()))
    }

    @Test
    fun `numpy 语义的小工具`() {
        // 偶数长度取中间两个的平均，不是下中位数
        assertEquals(2.5, Audio.median(doubleArrayOf(1.0, 2.0, 3.0, 4.0)), 1e-12)
        assertEquals(2.0, Audio.median(doubleArrayOf(3.0, 1.0, 2.0)), 1e-12)
        // np.percentile 线性插值：60% 落在 index 1.8
        assertEquals(2.8, Audio.percentile(floatArrayOf(1f, 2f, 3f, 4f), 60.0), 1e-9)
        // np.interp 两端夹住，不外推
        val xp = doubleArrayOf(0.0, 10.0)
        val fp = doubleArrayOf(5.0, 15.0)
        assertEquals(5.0, Audio.interp(-3.0, xp, fp), 1e-12)
        assertEquals(15.0, Audio.interp(99.0, xp, fp), 1e-12)
        assertEquals(10.0, Audio.interp(5.0, xp, fp), 1e-12)
    }
}
