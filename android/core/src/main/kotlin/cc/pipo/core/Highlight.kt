package cc.pipo.core

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * 回合分组与主题排序 —— ppai/highlight.py 的 Kotlin 移植。
 *
 * 这一层决定成片里出现哪几段。检测那层给的是一串时间点，
 * 这层把它们聚成回合、算出每个回合的力量，再按主题排序。
 *
 * 没有移植的部分，以及为什么：
 *
 *  * **发球四拍模板（serve_boundary）** —— 配置里是关闭的，而且是测过之后
 *    关掉的：三视频留一交叉验证 AUC 0.632，其中两个只有 0.492 / 0.534
 *    （随机水平）。根因是发球首拍触声常常检不出来，三个视频实际在测不同的
 *    物理事件。移过来只会把一个已知无效的东西带到新平台。
 *  * **画面运动（motion）** —— 只喂给「训练集锦」的综合评分，且需要逐帧解码。
 *    Android 首版不做：解码整段视频的代价远大于它对排序的贡献
 *    （权重 0.15，而另外三项都来自已经算好的音频量）。
 *    接口保留 motion 参数，将来要加不用改调用方。
 *
 * 几处必须和 numpy 对齐的语义（都在 Audio.kt 里有同样的注释）：
 *  * `ratios.std()` 是**总体**标准差（ddof=0），不是样本标准差
 *  * `np.percentile(av, 80)` 用线性插值
 *  * `np.median` 偶数长度取中间两个的平均
 */
object Highlight {

    data class Config(
        val gapS: Double = 0.7,
        val minDurationS: Double = 0.3,
        val minHits: Int = 2,
        val longestMinS: Double = 3.0,
        val trimBounce: Boolean = true,
        val bounceMinCount: Int = 4,
        val bounceFinalIoi: Double = 0.16,
        val bounceRatioStd: Double = 0.1,
        val bounceRatioLo: Double = 0.4,
        val bounceRatioHi: Double = 0.85,
        val bounceLookaheadS: Double = 3.0,
        val sparseGapS: Double = 4.0,
        val wRallyLength: Double = 0.35,
        val wPower: Double = 0.0,
        val wHitRate: Double = 0.15,
        val wMotion: Double = 0.15,
        val wDuration: Double = 0.10,
    )

    data class Rally(
        val start: Double,
        val end: Double,
        val hits: Int,
        var power: Double,        // 回合内较强击球的力量（80 分位，不是均值）
        val peakPower: Double,
        val tailPower: Double,
        val lastHit: Double,
        val endedWithBounce: Boolean,
        var score: Double = 0.0,
        var motion: Double = 0.0,
    ) {
        val duration: Double get() = end - start
    }

    data class Kind(val kind: String, val busy: Double, val medianGap: Double)

    /**
     * 找球落地后的连续弹跳，返回弹跳起点（回合真正结束的地方）。
     *
     * 物理依据：弹跳的恢复系数是常数，每次保留固定比例的能量，
     * 所以相邻间隔按**固定比值**收缩。只看「递减」会把真实回合切断 ——
     * 回合内的击球也会出现间隔递减，但比值是散的（标准差 0.19 vs 0.06）。
     * 不加这条判据的后果实测过：捡球画面被剪进集锦。
     */
    fun findBounceDecay(ts: DoubleArray, c: Config): Double? {
        if (ts.size < c.bounceMinCount + 1) return null
        val ioi = DoubleArray(ts.size - 1) { ts[it + 1] - ts[it] }
        var best: Double? = null
        for (i in 0..ioi.size - c.bounceMinCount) {
            var j = i
            while (j + 1 < ioi.size && ioi[j + 1] < ioi[j]) j++
            val n = j - i + 1
            if (n < c.bounceMinCount) continue
            if (ioi[j] > c.bounceFinalIoi) continue          // 弹跳末尾必然很密
            val m = n - 1
            val ratios = DoubleArray(m) { ioi[i + it + 1] / max(ioi[i + it], 1e-6) }
            val mean = ratios.sum() / m
            // numpy 的 std 默认 ddof=0（总体标准差）
            var v = 0.0
            for (r in ratios) v += (r - mean) * (r - mean)
            if (sqrt(v / m) > c.bounceRatioStd) continue      // 比值必须一致
            if (mean < c.bounceRatioLo || mean > c.bounceRatioHi) continue
            if (best == null || ts[i] < best!!) best = ts[i]
        }
        return best
    }

    /**
     * 把击球瞬态按间隔聚成活动片段。
     *
     * 注意用词：聚出来的**不是「回合」**。对着穷尽标注实测，真实回合只有
     * 0.5-2.8 秒、2-5 拍，而这里输出的片段是若干短回合加弹跳、噪声连成的
     * 活动密集区。对训练录像的集锦来说这反而合适，但 hits 不能理解成
     * 「这个回合打了多少拍」。
     */
    fun rallies(hits: DoubleArray, amps: DoubleArray?, c: Config): List<Rally> {
        if (hits.isEmpty()) return emptyList()
        val av = amps ?: DoubleArray(hits.size) { 1.0 }

        // 分组：与上一组最后一个瞬态的间隔超过 gap 就开新组
        val groups = ArrayList<MutableList<Int>>()
        groups.add(mutableListOf(0))
        for (i in 1 until hits.size) {
            val last = groups.last().last()
            if (hits[i] - hits[last] > c.gapS) groups.add(mutableListOf(i))
            else groups.last().add(i)
        }

        val out = ArrayList<Rally>()
        for (g in groups) {
            var ts = DoubleArray(g.size) { hits[g[it]] }
            var aa = DoubleArray(g.size) { av[g[it]] }
            var endedWithBounce = false

            if (c.trimBounce) {
                val b = findBounceDecay(ts, c)
                if (b != null && b > ts[0]) {
                    // 1) 回合内部混进了弹跳 —— 在弹跳起点截断
                    var k = 0
                    while (k < ts.size && ts[k] <= b) k++
                    ts = ts.copyOfRange(0, k)
                    aa = aa.copyOfRange(0, k)
                    endedWithBounce = true
                } else if (ts.isNotEmpty()) {
                    // 2) gap 降到 0.7 之后弹跳会被切成独立的簇，不再落在回合内部，
                    //    所以要往回合结束之后看：紧随其后的瞬态是不是弹跳衰减
                    val lastT = ts[ts.size - 1]
                    val after = ArrayList<Double>()
                    for (h in hits) if (h > lastT && h <= lastT + c.bounceLookaheadS) after.add(h)
                    if (after.size >= c.bounceMinCount) {
                        val probe = DoubleArray(after.size + 1)
                        probe[0] = lastT
                        for (i in after.indices) probe[i + 1] = after[i]
                        if (findBounceDecay(probe, c) != null) endedWithBounce = true
                    }
                }
            }
            if (ts.isEmpty()) continue

            out.add(Rally(
                start = ts[0], end = ts[ts.size - 1], hits = ts.size,
                // 力量取较强的那部分而不是均值：一个回合里总有轻挡和过渡球，
                // 用均值会把爆发力强的回合和平稳的回合拉平
                power = percentile(aa, 80.0),
                peakPower = aa.max(),
                // 收尾力量：最后两拍的最大值
                tailPower = aa.copyOfRange(max(0, aa.size - 2), aa.size).max(),
                lastHit = ts[ts.size - 1],
                endedWithBounce = endedWithBounce,
            ))
        }
        return out.filter { it.duration >= c.minDurationS && it.hits >= c.minHits }
    }

    /** 与 Python 的 rallies_gated 等价：发球门禁关闭时它就是 rallies。 */
    fun ralliesGated(hits: DoubleArray, amps: DoubleArray?, c: Config): List<Rally> =
        rallies(hits, amps, c)

    /**
     * 素材结构。稀疏型的价值在「删」（一半以上时间在捡球），
     * 密集型没什么可删，价值在「选」。只做二分 —— 再细分需要知道场上有几个人，
     * 那是视觉问题。
     */
    fun videoKind(rs: List<Rally>, duration: Double, c: Config): Kind {
        if (rs.isEmpty()) return Kind("unknown", 0.0, 0.0)
        val gaps = if (rs.size > 1)
            DoubleArray(rs.size - 1) { rs[it + 1].start - rs[it].end } else doubleArrayOf(0.0)
        val busy = rs.sumOf { it.duration } / max(duration, 1e-6)
        val mg = Audio.median(gaps.copyOf())
        return Kind(if (mg >= c.sparseGapS) "sparse" else "dense",
            Math.round(busy * 1000.0) / 1000.0, Math.round(mg * 100.0) / 100.0)
    }

    /** 综合评分。motionTimes/motionVals 为空时 motion 项恒为 0（Android 首版即如此）。 */
    fun score(rs: List<Rally>, motionTimes: DoubleArray, motionVals: DoubleArray,
              c: Config): List<Rally> {
        if (rs.isEmpty()) return rs
        val n = rs.size
        val nHits = DoubleArray(n) { rs[it].hits.toDouble() }
        val dur = DoubleArray(n) { rs[it].duration }
        val power = DoubleArray(n) { rs[it].power }
        val rate = DoubleArray(n) { nHits[it] / max(dur[it], 1e-6) }
        val motion = DoubleArray(n) { i ->
            if (motionTimes.size > 1) {
                var s = 0.0; var k = 0
                for (j in motionTimes.indices)
                    if (motionTimes[j] >= rs[i].start && motionTimes[j] <= rs[i].end) {
                        s += motionVals[j]; k++
                    }
                if (k > 0) s / k else 0.0
            } else 0.0
        }
        fun nz(x: DoubleArray): DoubleArray {
            val lo = x.min(); val hi = x.max()
            return if (hi > lo) DoubleArray(x.size) { (x[it] - lo) / (hi - lo) } else DoubleArray(x.size)
        }
        val a = nz(nHits); val b = nz(dur); val d = nz(rate); val e = nz(motion); val f = nz(power)
        for (i in 0 until n) {
            val v = c.wRallyLength * a[i] + c.wDuration * b[i] + c.wHitRate * d[i] +
                    c.wMotion * e[i] + c.wPower * f[i]
            rs[i].score = pyRound(v, 4)
            rs[i].motion = pyRound(motion[i], 3)
            // Python 的 score() 在这里把 power 收成 1 位小数，下游（包括排序里
            // peak_power 缺失时的兜底）看到的就是这个收过的值。不跟着收，
            // 两边会差最多 0.05 —— 单看不大，但排序并列时会翻到不同的一段。
            rs[i].power = pyRound(rs[i].power, 1)
        }
        return rs
    }

    /**
     * Python 的 round()：**半数取偶**（银行家舍入），不是半数进一。
     * `round(2.5) == 2` 而 `Math.round(2.5) == 3`。这个项目里所有对外的数
     * 都过一次 round，两边规则不同就会零星地差一个末位。
     */
    internal fun pyRound(v: Double, digits: Int): Double =
        java.math.BigDecimal(v).setScale(digits, java.math.RoundingMode.HALF_EVEN).toDouble()

    /**
     * 按主题排序。
     *
     * 下线的主题（kill 最帅击球 / weak 失误合集 / records 精彩瞬间）没有移植 ——
     * 它们都是**测过之后**去掉的：前两个用的 tail_power 只有 48% 取到真挥拍，
     * records 与「扣杀瞬间」「最长相持」各自的第一名完全重合（8 个视频验证）。
     */
    fun rank(rs: List<Rally>, kind: String, c: Config): List<Rally> = when (kind) {
        // 时长下限：短于 3 秒的不叫相持。多球训练素材前十原来是 2.1-3.5 秒，
        // 贴「最长相持」这个名字名不副实。
        "longest" -> rs.filter { it.duration >= c.longestMinS }
            .sortedWith(compareByDescending<Rally> { it.hits }.thenByDescending { it.duration })
        // 密集素材里回合长度都差不多，能拉开差距的是单拍的绝对力量
        "power" -> rs.sortedByDescending { if (it.peakPower != 0.0) it.peakPower else it.power }
        else -> rs.sortedByDescending { it.score }
    }

    /** np.percentile 的默认线性插值法（会就地排序副本）。 */
    internal fun percentile(x: DoubleArray, q: Double): Double {
        if (x.isEmpty()) return 0.0
        val a = x.copyOf()
        a.sort()
        val pos = (q / 100.0) * (a.size - 1)
        val lo = Math.floor(pos).toInt()
        val hi = Math.ceil(pos).toInt()
        if (lo == hi) return a[lo]
        return a[lo] + (a[hi] - a[lo]) * (pos - lo)
    }
}
