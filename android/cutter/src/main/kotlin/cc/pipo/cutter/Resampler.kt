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

        // 截止取上下采样率里小的那个的一半（防混叠），归一化到上采样后的速率。
        // 滤波器构造抽成 buildFilter —— **流式版 [Stream] 必须用同一份**，
        // 两边只要差一个系数，逐位相同就无从谈起。
        val tapsPerPhase = HALF_TAPS * 2
        val filterLen = tapsPerPhase * up
        val h = buildFilter(up, down, filterLen)

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

    /**
     * 流式重采样 —— **和 [resample] 逐位相同，但不需要整段音频在内存里。**
     *
     * 为什么必须有这个
     * ----------------
     * 原来的解码是「整段按原始采样率攒成一堆小数组，再另开一个同样大的
     * 数组拼起来，最后才降采样」。12 分钟 48kHz 单声道就是
     * 727 秒 × 48000 × 4 字节 = **139MB**，而拼接那一步要同时拿着两份，
     * 峰值 279MB —— 一台真实用户的手机就死在这里：
     *
     *     Failed to allocate a 139739088 byte allocation
     *     with 25100288 free bytes and 55MB until OOM
     *
     * 而降采样之后只有 46MB（16kHz）。也就是说**那 279MB 全是中转开销**，
     * 一秒钟都不需要存在。我们手上所有测试素材都是 14-60 秒（最多 11MB），
     * 所以这条路一次都没走到过。
     *
     * 为什么可以严格等价
     * ------------------
     * 这个滤波器的支持区间是有限的：输出第 m 点只用到输入的
     * `[start-15, start+16]`（[HALF_TAPS] 和 tapsPerPhase 决定），
     * 其中 `start = m*down/up`。所以只要留住那一小段历史，
     * 分块喂和整段喂算出来的每个点**累加顺序都一样**，结果逐位相同 ——
     * 这一条由 ResamplerStreamTest 用随机数据和不规则分块实测钉住。
     *
     * 边界也一样：越界的抽头按 0 处理，和 [resample] 里的
     * `if (idx in x.indices)` 是同一个语义。
     */
    class Stream(private val inRate: Int, private val outRate: Int) {

        private val g = gcd(inRate, outRate)
        private val up = outRate / g
        private val down = inRate / g
        private val tapsPerPhase = HALF_TAPS * 2
        private val filterLen = tapsPerPhase * up
        private val h: DoubleArray = buildFilter(up, down, filterLen)

        /** 已经喂进来的输入采样总数。 */
        private var nIn = 0L
        /** 下一个要产出的输出下标。 */
        private var m = 0L
        /**
         * 输入的滑动窗口。[bufStart] 是 buf[0] 对应的全局输入下标。
         * 只留还可能被用到的那一小段，其余随时丢掉。
         */
        private var buf = FloatArray(4096)
        private var bufLen = 0
        private var bufStart = 0L

        /** 直通（同采样率）时不做任何事，调用方直接拿原始数据。 */
        val passthrough: Boolean get() = inRate == outRate

        private fun ensure(extra: Int) {
            if (bufLen + extra <= buf.size) return
            var cap = buf.size
            while (cap < bufLen + extra) cap *= 2
            buf = buf.copyOf(cap)
        }

        /** 丢掉不会再被用到的历史。 */
        private fun compact() {
            // 下一个输出点最早会读到 start(m) - (tapsPerPhase-1) + HALF_TAPS
            val need = (m * down / up) - (tapsPerPhase - 1) + HALF_TAPS
            val drop = (need - bufStart).coerceAtLeast(0L).coerceAtMost(bufLen.toLong()).toInt()
            if (drop <= 0) return
            System.arraycopy(buf, drop, buf, 0, bufLen - drop)
            bufLen -= drop
            bufStart += drop
        }

        private fun emitOne(): Float {
            val t = m * down
            val phase = (t % up).toInt()
            val start = t / up
            var acc = 0.0
            for (k in 0 until tapsPerPhase) {
                val tap = phase + k * up
                if (tap >= filterLen) break
                val idx = start - k + HALF_TAPS
                if (idx in 0 until nIn) {
                    val local = (idx - bufStart).toInt()
                    if (local in 0 until bufLen) acc += h[tap] * buf[local]
                }
            }
            m++
            return acc.toFloat()
        }

        /**
         * 喂一段输入，把此刻已经能确定的输出写进 [sink]。
         *
         * 「能确定」= 该点用到的最靠后的输入（`start + HALF_TAPS`）已经到了。
         */
        fun feed(x: FloatArray, n: Int, sink: (Float) -> Unit) {
            if (n <= 0) return
            compact()
            ensure(n)
            System.arraycopy(x, 0, buf, bufLen, n)
            bufLen += n
            nIn += n
            // start(m) + HALF_TAPS <= nIn - 1
            while ((m * down / up) + HALF_TAPS <= nIn - 1) {
                sink(emitOne())
                if (bufLen > 1 shl 16) compact()
            }
        }

        /** 输入结束。补齐尾巴上那几个点 —— 越界抽头按 0，和整段版一致。 */
        fun finish(sink: (Float) -> Unit) {
            val outLen = ceil(nIn.toDouble() * up / down).toLong()
            while (m < outLen) sink(emitOne())
        }
    }

    /** [resample] 和 [Stream] 共用的滤波器构造，两边必须完全一致。 */
    private fun buildFilter(up: Int, down: Int, filterLen: Int): DoubleArray {
        val fc = CUTOFF * 0.5 / maxOf(up, down)
        val mid = filterLen / 2.0
        val h = DoubleArray(filterLen)
        val i0beta = bessel0(BETA)
        for (n in 0 until filterLen) {
            val t = n - mid
            val s = if (abs(t) < 1e-9) 2.0 * fc else sin(2.0 * PI * fc * t) / (PI * t)
            val r = 2.0 * n / (filterLen - 1) - 1.0
            val w = bessel0(BETA * Math.sqrt(maxOf(0.0, 1.0 - r * r))) / i0beta
            h[n] = s * w * up
        }
        return h
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
