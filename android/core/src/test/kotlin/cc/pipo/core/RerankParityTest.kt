package cc.pipo.core

import java.io.DataInputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Properties
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * 重排推理对着 Python 逐点比对，跑在 JVM 上。
 *
 * 输入用 Python 已经抽好的 32kHz PCM，所以这里量的**只有 ONNX 推理和
 * 逻辑回归**这两步 —— 解码和重采样的差异在 AudioDecodeParityTest 里单独量过
 * （92.9% 击球匹配、时间戳零偏差），两件事分开量才知道偏差出自哪。
 *
 * 需要 models/cnn14.onnx（323MB，fp32）。没有就跳过而不是失败：
 * 它不进版本库、不进安装包，产品里是「第一次点剪辑时按需下载」的那份。
 */
class RerankParityTest {

    // Gradle 跑测试时工作目录是模块目录（android/core），仓库根在它上面两级。
    // 之前只往上一级，结果四个实质测试全走了「模型不存在」的跳过分支，
    // 报告里显示 PASSED —— 这种「静默跳过」比失败更危险，所以下面
    // 把找不到模型时的路径也打出来。
    private val repo = File(System.getProperty("user.dir")).parentFile.parentFile
    private val onnx = File(repo, "models/cnn14.onnx")
    private val weights = File(repo, "models/rerank.bin")

    private fun res(name: String): ByteArray =
        RerankParityTest::class.java.getResourceAsStream("/" + name)!!
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
        RerankParityTest::class.java.getResourceAsStream("/rr.properties")!!.use { load(it) }
    }

    private fun available(): Boolean {
        if (!onnx.exists()) {
            println("  跳过：找不到 ${onnx.absolutePath}（323MB 模型不进版本库）")
            return false
        }
        return true
    }

    @Test
    fun `权重文件读得对`() {
        assertTrue(weights.exists(), "缺 ${weights.absolutePath}")
        val (coef, b) = Rerank.readWeights(weights)
        assertEquals(2048, coef.size, "系数维度")
        // **不写死截距** —— 它每次重训都会合法地变（实测从 0.0399 变成 -0.0292）。
        // 写死等于每加一份标注就红一次，那种测试很快会被当噪音关掉。
        // 对着导出时一起生成的基准比：真正要守的不变量是
        // 「rerank.bin 是 rerank.npz 的忠实导出」，不是某个具体数值。
        val want = p.getProperty("intercept")!!.toFloat()
        assertTrue(abs(b - want) < 1e-5, "截距 $b，导出时是 $want")
        assertTrue(coef.any { it != 0f }, "系数不该全为零")
        println("  权重 ${coef.size} 维，截距 $b")
    }

    @Test
    fun `嵌入与 Python 逐元素一致`() {
        if (!available()) return
        Rerank.load(onnx.absolutePath, weights.absolutePath).use { m ->
            val pcm = f32("rr_pcm32k.f32")
            val (times, feats) = Rerank.embed(pcm, m)
            val wantN = p.getProperty("nWindows")!!.toInt()
            assertEquals(wantN, times.size, "窗数")

            val wantFeat = f32("rr_feat0.f32")     // Python 的第 0 个窗的 2048 维
            var worst = 0.0
            for (j in wantFeat.indices) {
                worst = maxOf(worst, abs(feats[0][j] - wantFeat[j]).toDouble())
            }
            // 同一个 ONNX 图、同样的 fp32 权重，差别只该来自不同 BLAS 的累加顺序
            assertTrue(worst < 1e-3, "第 0 窗嵌入最大绝对差 $worst")
            println("  嵌入 ${times.size} 窗 × 2048，第 0 窗最大绝对差 %.2e".format(worst))
        }
    }

    @Test
    fun `候选概率与 Python 一致`() {
        if (!available()) return
        Rerank.load(onnx.absolutePath, weights.absolutePath).use { m ->
            val pcm = f32("rr_pcm32k.f32")
            val hits = f64("rr_hits.f64")
            val (times, feats) = Rerank.embed(pcm, m)
            val want = f64("rr_probs.f64")
            assertEquals(want.size, hits.size, "候选数")

            var worst = 0.0
            for (i in hits.indices) {
                var best = 0
                var bd = Double.MAX_VALUE
                for (j in times.indices) {
                    val d = abs(times[j] - hits[i])
                    if (d < bd) { bd = d; best = j }
                }
                worst = maxOf(worst, abs(Rerank.probability(feats[best], m) - want[i]))
            }
            assertTrue(worst < 1e-3, "概率最大差 $worst")
            println("  ${hits.size} 个候选，概率最大差 %.2e".format(worst))
        }
    }

    @Test
    fun `保留下来的候选与 Python 完全相同`() {
        if (!available()) return
        Rerank.load(onnx.absolutePath, weights.absolutePath).use { m ->
            val pcm = f32("rr_pcm32k.f32")
            val hits = f64("rr_hits.f64")
            val (times, feats) = Rerank.embed(pcm, m)
            val kept = Rerank.apply(hits, times, feats, m,
                p.getProperty("keepRatio")!!.toDouble())
            val want = f64("rr_kept.f64")
            assertEquals(want.size, kept.size, "保留数量")
            // 输入 PCM 相同、模型相同，这里应当**完全相同**，不给容差 ——
            // 这一步选错一个候选，回合边界就会跟着变。
            for (i in want.indices) {
                assertTrue(abs(kept[i] - want[i]) < 1e-9,
                    "第 $i 个：Python ${want[i]}，Kotlin ${kept[i]}")
            }
            println("  保留 ${kept.size}/${hits.size} 个，与 Python 逐点相同")
        }
    }

    @Test
    fun `全片平均概率与 Python 一致`() {
        if (!available()) return
        Rerank.load(onnx.absolutePath, weights.absolutePath).use { m ->
            val (_, feats) = Rerank.embed(f32("rr_pcm32k.f32"), m)
            val got = Rerank.meanProbability(feats, m)
            val want = p.getProperty("meanProb")!!.toDouble()
            // 这个数是「是不是乒乓球录像」的判据：阴性 0.009-0.010、
            // 真实素材 0.379-0.672，差 40 倍。必须对得很准才敢用同一个门槛。
            assertTrue(abs(got - want) < 1e-3, "平均概率 $got vs Python $want")
            println("  全片平均概率 %.6f（Python %.6f）".format(got, want))
        }
    }
}
