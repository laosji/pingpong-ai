package cc.pipo.cutter

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.min
import kotlin.math.sin

/**
 * 多相 windowed-sinc 重采样。
 *
 * 为什么不能用线性插值凑合
 * ------------------------
 * 桌面版是 ffmpeg 抽 PCM（内部走 swresample，Kaiser 窗 sinc 滤波）。
 * 检测靠的是**谱通量**——频域上的突变。线性插值等于叠了一个很差的低通，
 * 高频被压掉、还会折返出混叠分量，而击球声的能量恰好集中在 1-5kHz。
 * 换句话说，重采样算法选错，检测结果就跟着变，前面辛辛苦苦做的逐点一致
 * 会在这一步全部作废。
 *
 * 手机录像基本都是 48kHz：→16k 是整 3 倍抽取，→32k 是 2:3 有理比，
 * 所以用多相结构（按 L 上采样、按 M 抽取，只算真正需要的那些点）。
 *
 * 参数取得和 ffmpeg 的默认接近：截止频率取奈奎斯特的 0.97，
 * Kaiser β=8（约 -80dB 阻带）。**不追求逐位相同** —— 那要复刻 swr 的全部
 * 细节，成本远大于收益。追求的是「检出的击球时刻不变」，
 * 这一条由 ResampleParityTest 对着 Python 的输出实测。
 */
object Resampler {

    private const val CUTOFF = 0.97
    private const val BETA = 8.0
    private const val HALF_TAPS = 16      // 每相 32 抽头，和 ffmpeg 默认同量级

    /** 最大公约数 —— 把 48000/16000 化简成 1/3，相数才不会爆。 */
    private fun gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)

    private fun bessel0(x: Double): Double {
        // 零阶修正贝塞尔函数的级数展开。收敛很快，25 项足够到双精度。
        var sum = 1.0
        var term = 1.0
        val h = x / 2.0
        for (k in 1..25) {
            term *= (h / k) * (h / k)
            sum += term
            if (term < 1e-16 * sum) break
        }
        return sum
    }

    /**
     * [x] 从 [inRate] 重采样到 [outRate]。同采样率直接原样返回。
     */
    fun resample(x: FloatArray, inRate: Int, outRate: Int): FloatArray {
        if (inRate == outRate || x.isEmpty()) return x
        val g = gcd(inRate, outRate)
        val up = outRate / g          // L
        val down = inRate / g         // M

        // 截止取上下采样率里小的那个的一半（防混叠），归一化到上采样后的速率
        val fc = CUTOFF * 0.5 / maxOf(up, down)
        val tapsPerPhase = HALF_TAPS * 2
        val filterLen = tapsPerPhase * up
        val mid = filterLen / 2.0

        // 预生成整条滤波器，再按相拆开
        val h = DoubleArray(filterLen)
        val i0beta = bessel0(BETA)
        for (n in 0 until filterLen) {
            val t = n - mid
            val s = if (abs(t) < 1e-9) 2.0 * fc else sin(2.0 * PI * fc * t) / (PI * t)
            // Kaiser 窗
            val r = 2.0 * n / (filterLen - 1) - 1.0
            val w = bessel0(BETA * Math.sqrt(maxOf(0.0, 1.0 - r * r))) / i0beta
            h[n] = s * w * up     // 乘 up 补上采样带来的能量损失
        }

        val outLen = ceil(x.size.toDouble() * up / down).toInt()
        val out = FloatArray(outLen)
        for (m in 0 until outLen) {
            // 输出第 m 点对应上采样序列的第 m*down 点
            val t = m.toLong() * down
            val phase = (t % up).toInt()
            val start = (t / up).toInt()
            var acc = 0.0
            // 只取该相的抽头：h[phase + k*up]
            for (k in 0 until tapsPerPhase) {
                val tap = phase + k * up
                if (tap >= filterLen) break
                val idx = start - k + HALF_TAPS
                if (idx in x.indices) acc += h[tap] * x[idx]
            }
            out[m] = acc.toFloat()
        }
        return out
    }

    /** 多声道交织 PCM 混成单声道 —— 检测只用单声道。 */
    fun toMono(interleaved: FloatArray, channels: Int): FloatArray {
        if (channels <= 1) return interleaved
        val n = interleaved.size / channels
        val out = FloatArray(n)
        for (i in 0 until n) {
            var s = 0.0
            for (c in 0 until channels) s += interleaved[i * channels + c]
            out[i] = (s / channels).toFloat()
        }
        return out
    }

    internal fun minOfInt(a: Int, b: Int) = min(a, b)
}
