package cc.pipo.core

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.ln1p
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * 击球声检测 —— ppai/audio.py 的 Kotlin 移植。
 *
 * 这是整个产品的地基：回合分组、主题排序、成片时长全都建在它输出的
 * 那串时间点上。所以它**必须和 Python 逐点一致**，不是「差不多就行」。
 * 同目录的 AudioParityTest 拿 Python 在真实素材上的输出做基准逐点比对。
 *
 * 移植时几处必须和 numpy 对齐的细节（都踩过或差点踩到）：
 *
 *  * `np.hanning(N)` 是**对称**窗 `0.5 - 0.5*cos(2πn/(N-1))`，
 *    不是信号处理里常见的周期窗 `2πn/N`。差一个样本，谱就对不上。
 *  * `np.median` 在偶数长度上取中间两个的**平均**，不是下中位数。
 *  * `np.percentile(x, 60)` 用线性插值（numpy 默认），不是取最近秩。
 *  * `np.interp` 在两端**夹住**（用首/末值），不外推。
 *  * 峰值判据左闭右开：`env[i] >= env[i-1] && env[i] > env[i+1]`，
 *    两边都用 `>` 会漏掉平顶峰，都用 `>=` 会把平台段整段选中。
 *
 * 不依赖 Android，纯 JVM —— 这样单测不用模拟器就能跑，改一行五秒钟知道对错。
 */
object Audio {

    data class Config(
        val sr: Int = 16000,
        val nFft: Int = 512,
        val hop: Int = 160,
        val fmin: Double = 800.0,
        val fmax: Double = 6000.0,
        val kMad: Double = 4.0,
        val noiseWinS: Double = 2.0,
        val minGapS: Double = 0.06,
    )

    data class Result(
        val hits: DoubleArray,      // 击球时刻（秒）
        val env: FloatArray,        // 谱通量包络
        val thr: FloatArray,        // 自适应阈值曲线
        val frameRate: Double,
    )

    /** 谱通量包络。分块处理，一小时素材也只占几 MB。 */
    fun onsetEnvelope(x: FloatArray, c: Config): Pair<FloatArray, Double> {
        val frameRate = c.sr.toDouble() / c.hop
        if (x.size < c.nFft) return FloatArray(0) to frameRate

        // 对称 Hann，和 np.hanning 一致
        val win = FloatArray(c.nFft) {
            (0.5 - 0.5 * cos(2.0 * Math.PI * it / (c.nFft - 1))).toFloat()
        }
        // rfftfreq(nFft, 1/sr)[k] = k * sr / nFft
        val nBins = c.nFft / 2 + 1
        var lo = -1
        var hi = -1
        for (k in 0 until nBins) {
            val f = k.toDouble() * c.sr / c.nFft
            if (f >= c.fmin && f <= c.fmax) {
                if (lo < 0) lo = k
                hi = k
            }
        }
        if (lo < 0) return FloatArray(0) to frameRate
        val bandLen = hi - lo + 1

        val nFrames = 1 + (x.size - c.nFft) / c.hop
        val env = FloatArray(nFrames)
        val fft = Fft(c.nFft)
        val re = DoubleArray(c.nFft)
        val im = DoubleArray(c.nFft)
        var prev: DoubleArray? = null
        val cur = DoubleArray(bandLen)

        for (t in 0 until nFrames) {
            val off = t * c.hop
            for (i in 0 until c.nFft) {
                re[i] = (x[off + i] * win[i]).toDouble()
                im[i] = 0.0
            }
            fft.transform(re, im)
            for (k in 0 until bandLen) {
                val b = lo + k
                val mag = sqrt(re[b] * re[b] + im[b] * im[b])
                cur[k] = ln1p(mag * 100.0)   // 对数压缩：只看相对突变，不看绝对音量
            }
            // 第一帧没有前一帧，Python 用它自己当 head，所以通量恒为 0
            val p = prev
            var sum = 0.0
            if (p != null) {
                for (k in 0 until bandLen) {
                    val d = cur[k] - p[k]
                    if (d > 0) sum += d          // 只要「突然出现的能量」，不要衰减
                }
            }
            env[t] = sum.toFloat()
            prev = cur.copyOf()
        }
        return env to frameRate
    }

    /** 局部中位数 + k·MAD。按块算再线性插值，避免 O(N·W) 的滑动中位数。 */
    fun adaptiveThreshold(env: FloatArray, frameRate: Double, winS: Double, k: Double): FloatArray {
        val n = env.size
        if (n == 0) return FloatArray(0)
        val w = max(1, (winS * frameRate).roundToInt())
        val nBlocks = max(1, Math.ceil(n.toDouble() / w).toInt())
        val padded = DoubleArray(nBlocks * w) { if (it < n) env[it].toDouble() else env[n - 1].toDouble() }

        val med = DoubleArray(nBlocks)
        val mad = DoubleArray(nBlocks)
        val buf = DoubleArray(w)
        for (b in 0 until nBlocks) {
            System.arraycopy(padded, b * w, buf, 0, w)
            val m = median(buf.copyOf())
            med[b] = m
            for (i in 0 until w) buf[i] = abs(padded[b * w + i] - m)
            mad[b] = median(buf.copyOf()) * 1.4826
        }

        val centers = DoubleArray(nBlocks) { it * w + w / 2.0 }
        // MAD 可能为 0（极安静段），给下限，否则阈值塌到 0 会满屏误检
        val floor = percentile(env, 60.0) * 0.05
        return FloatArray(n) { i ->
            val m = interp(i.toDouble(), centers, med)
            val a = interp(i.toDouble(), centers, mad)
            (m + k * max(a, floor)).toFloat()
        }
    }

    fun detectHits(x: FloatArray, c: Config): Result {
        val (env, frameRate) = onsetEnvelope(x, c)
        if (env.isEmpty()) return Result(DoubleArray(0), env, FloatArray(0), frameRate)
        val thr = adaptiveThreshold(env, frameRate, c.noiseWinS, c.kMad)

        // 局部极大值且超过阈值
        val cand = ArrayList<Int>()
        for (i in 1 until env.size - 1) {
            if (env[i] >= env[i - 1] && env[i] > env[i + 1] && env[i] > thr[i]) cand.add(i)
        }
        if (cand.isEmpty()) return Result(DoubleArray(0), env, thr, frameRate)

        // 按强度降序贪心去重，保证最小间隔。
        // 并列时按帧号升序 —— numpy 的 argsort 默认是 quicksort（不稳定），
        // 真出现完全相同的通量值时两边可能选不同的那个。实测基准里没有并列，
        // 这里定死顺序是为了让 Kotlin 侧自己可复现。
        val minGap = max(1, (c.minGapS * frameRate).roundToInt())
        val order = cand.sortedWith(compareByDescending<Int> { env[it] }.thenBy { it })
        val taken = BooleanArray(env.size)
        val kept = ArrayList<Int>()
        for (i in order) {
            val a = max(0, i - minGap)
            val b = min(env.size, i + minGap + 1)
            var clash = false
            for (j in a until b) if (taken[j]) { clash = true; break }
            if (!clash) { kept.add(i); taken[i] = true }
        }
        kept.sort()
        return Result(DoubleArray(kept.size) { kept[it] / frameRate }, env, thr, frameRate)
    }

    /** 每次击球的通量峰值 —— 近似「力量」。取峰值附近最大值，容忍几帧定位误差。 */
    fun hitAmplitudes(hits: DoubleArray, env: FloatArray, frameRate: Double): DoubleArray {
        if (env.isEmpty() || hits.isEmpty()) return DoubleArray(hits.size)
        return DoubleArray(hits.size) { n ->
            val i = (hits[n] * frameRate).toInt().coerceIn(0, env.size - 1)
            var m = Double.NEGATIVE_INFINITY
            for (j in max(0, i - 2)..min(env.size - 1, i + 2)) m = max(m, env[j].toDouble())
            m
        }
    }

    // ── numpy 语义的小工具 ─────────────────────────────────

    /** np.median：偶数长度取中间两个的平均。会就地排序传入的数组。 */
    internal fun median(a: DoubleArray): Double {
        if (a.isEmpty()) return 0.0
        a.sort()
        val n = a.size
        return if (n % 2 == 1) a[n / 2] else (a[n / 2 - 1] + a[n / 2]) / 2.0
    }

    /** np.percentile 的默认线性插值法。 */
    internal fun percentile(x: FloatArray, q: Double): Double {
        if (x.isEmpty()) return 0.0
        val a = DoubleArray(x.size) { x[it].toDouble() }
        a.sort()
        val pos = (q / 100.0) * (a.size - 1)
        val lo = Math.floor(pos).toInt()
        val hi = Math.ceil(pos).toInt()
        if (lo == hi) return a[lo]
        return a[lo] + (a[hi] - a[lo]) * (pos - lo)
    }

    /** np.interp：xp 递增，两端夹住不外推。 */
    internal fun interp(v: Double, xp: DoubleArray, fp: DoubleArray): Double {
        if (xp.isEmpty()) return 0.0
        if (xp.size == 1 || v <= xp[0]) return fp[0]
        if (v >= xp[xp.size - 1]) return fp[fp.size - 1]
        var lo = 0
        var hi = xp.size - 1
        while (hi - lo > 1) {
            val mid = (lo + hi) ushr 1
            if (xp[mid] <= v) lo = mid else hi = mid
        }
        val t = (v - xp[lo]) / (xp[hi] - xp[lo])
        return fp[lo] + t * (fp[hi] - fp[lo])
    }
}

/**
 * 迭代基-2 FFT。nFft 是 512（2 的幂），所以够用。
 *
 * 不引第三方 FFT 库：这是唯一需要的信号处理原语，为它加一个依赖
 * 会把「core 模块不依赖任何东西」这条破掉，而那条是它能被 JVM 单测
 * 直接验证的前提。
 */
internal class Fft(private val n: Int) {
    private val levels = Integer.numberOfTrailingZeros(n)
    private val cosT = DoubleArray(n / 2) { cos(2.0 * Math.PI * it / n) }
    private val sinT = DoubleArray(n / 2) { Math.sin(2.0 * Math.PI * it / n) }

    init {
        require(n > 0 && Integer.bitCount(n) == 1) { "FFT 长度必须是 2 的幂，收到 $n" }
    }

    fun transform(re: DoubleArray, im: DoubleArray) {
        // 位反转置换
        for (i in 0 until n) {
            val j = Integer.reverse(i) ushr (32 - levels)
            if (j > i) {
                var t = re[i]; re[i] = re[j]; re[j] = t
                t = im[i]; im[i] = im[j]; im[j] = t
            }
        }
        var size = 2
        while (size <= n) {
            val half = size / 2
            val step = n / size
            var i = 0
            while (i < n) {
                var j = i
                var k = 0
                while (j < i + half) {
                    val l = j + half
                    val tre = re[l] * cosT[k] + im[l] * sinT[k]
                    val tim = -re[l] * sinT[k] + im[l] * cosT[k]
                    re[l] = re[j] - tre; im[l] = im[j] - tim
                    re[j] += tre; im[j] += tim
                    j++; k += step
                }
                i += size
            }
            size *= 2
        }
    }
}
