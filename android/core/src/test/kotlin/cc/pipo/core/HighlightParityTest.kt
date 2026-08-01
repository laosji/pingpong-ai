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
 * 回合分组与排序对着 Python 逐条比对。
 *
 * 输入是 Python 已经算好的击球时刻和力量（不是 PCM），所以基准可以覆盖
 * **整段 7.4 分钟素材**（971 个瞬态 → 85 个回合），而不是只有 20 秒。
 * 这层的 bug 恰恰只在长素材上才露出来 —— 弹跳收尾、稀疏/密集判定、
 * 排序并列，都要有足够多的回合才碰得到。
 */
class HighlightParityTest {

    private fun res(name: String) =
        HighlightParityTest::class.java.getResourceAsStream("/" + name)!!
            .use { DataInputStream(it).readBytes() }

    private fun f64(name: String): DoubleArray {
        val b = ByteBuffer.wrap(res(name)).order(ByteOrder.LITTLE_ENDIAN)
        return DoubleArray(b.remaining() / 8) { b.double }
    }

    private val p = Properties().apply {
        HighlightParityTest::class.java.getResourceAsStream("/highlight.properties")!!
            .use { load(it) }
    }

    private fun s(k: String) = p.getProperty(k)!!

    private val cfg = Highlight.Config(
        gapS = s("gapS").toDouble(),
        minDurationS = s("minDurationS").toDouble(),
        minHits = s("minHits").toInt(),
        longestMinS = s("longestMinS").toDouble(),
        trimBounce = s("trimBounce").toBoolean(),
        bounceMinCount = s("bounceMinCount").toInt(),
        bounceFinalIoi = s("bounceFinalIoi").toDouble(),
        bounceRatioStd = s("bounceRatioStd").toDouble(),
        bounceRatioLo = s("bounceRatioLo").toDouble(),
        bounceRatioHi = s("bounceRatioHi").toDouble(),
        bounceLookaheadS = s("bounceLookaheadS").toDouble(),
        sparseGapS = s("sparseGapS").toDouble(),
        wRallyLength = s("wRallyLength").toDouble(),
        wPower = s("wPower").toDouble(),
        wHitRate = s("wHitRate").toDouble(),
        wMotion = s("wMotion").toDouble(),
        wDuration = s("wDuration").toDouble(),
    )

    private val hits = f64("rally_hits.f64")
    private val amps = f64("rally_amps.f64")
    private val duration = s("duration").toDouble()

    /** 只取需要的字段，不为一个测试引 JSON 库 —— core 模块保持零依赖。 */
    private val json = String(res("rally_expected.json"), Charsets.UTF_8)

    private fun nums(field: String): List<Double> {
        val i = json.indexOf("\"$field\":")
        require(i >= 0) { "基准里没有 $field" }
        val a = json.indexOf('[', i)
        val b = json.indexOf(']', a)
        val body = json.substring(a + 1, b).trim()
        if (body.isEmpty()) return emptyList()
        return body.split(",").map { it.trim().toDouble() }
    }

    private fun objField(name: String): List<String> {
        // "rallies":[{...},{...}] —— 按对象切开
        val i = json.indexOf("\"rallies\":")
        val a = json.indexOf('[', i)
        val out = ArrayList<String>()
        var depth = 0
        var start = -1
        for (k in a until json.length) {
            when (json[k]) {
                '{' -> { if (depth == 0) start = k; depth++ }
                '}' -> { depth--; if (depth == 0) out.add(json.substring(start, k + 1)) }
                ']' -> if (depth == 0) return out
            }
        }
        return out
    }

    private fun field(obj: String, k: String): Double {
        val i = obj.indexOf("\"$k\":")
        require(i >= 0) { "对象里没有 $k：$obj" }
        val v = obj.substring(i + k.length + 3).takeWhile { it != ',' && it != '}' }.trim()
        return if (v == "true") 1.0 else if (v == "false") 0.0 else v.toDouble()
    }

    private val expected = objField("rallies")

    @Test
    fun `回合数量与边界一致`() {
        val got = Highlight.ralliesGated(hits, amps, cfg)
        assertEquals(expected.size, got.size,
            "回合数不一致：Python ${expected.size}，Kotlin ${got.size}")
        for (i in expected.indices) {
            val e = expected[i]
            assertTrue(abs(got[i].start - field(e, "start")) < 1e-6,
                "第 $i 个回合起点：Python ${field(e, "start")}，Kotlin ${got[i].start}")
            assertTrue(abs(got[i].end - field(e, "end")) < 1e-6, "第 $i 个回合终点")
            assertEquals(field(e, "hits").toInt(), got[i].hits, "第 $i 个回合的瞬态数")
        }
        println("  回合 ${got.size} 个，边界与瞬态数全部一致")
    }

    @Test
    fun `力量与弹跳收尾一致`() {
        // 必须先跑 score()：Python 在那里把 power 收成 1 位小数，
        // 基准也是 score 之后导出的。直接比 ralliesGated 的原始值会差最多 0.05。
        val got = Highlight.score(
            Highlight.ralliesGated(hits, amps, cfg), DoubleArray(0), DoubleArray(0), cfg)
        var bounce = 0
        for (i in expected.indices) {
            val e = expected[i]
            assertTrue(abs(got[i].power - field(e, "power")) < 1e-4, "第 $i 个回合的力量")
            assertTrue(abs(got[i].peakPower - field(e, "peak_power")) < 1e-4, "第 $i 个峰值力量")
            assertTrue(abs(got[i].tailPower - field(e, "tail_power")) < 1e-4, "第 $i 个收尾力量")
            val wantB = field(e, "ended_with_bounce") > 0.5
            assertEquals(wantB, got[i].endedWithBounce, "第 $i 个回合的弹跳收尾判定")
            if (wantB) bounce++
        }
        println("  力量三项一致；弹跳收尾 $bounce 个，判定一致")
    }

    @Test
    fun `素材结构判定一致`() {
        val got = Highlight.ralliesGated(hits, amps, cfg)
        val k = Highlight.videoKind(got, duration, cfg)
        val wantKind = json.substringAfter("\"kind\":\"").substringBefore("\"")
        val wantBusy = json.substringAfter("\"busy\":").substringBefore(",").toDouble()
        val wantGap = json.substringAfter("\"median_gap\":").substringBefore(",").toDouble()
        assertEquals(wantKind, k.kind, "结构类型")
        assertTrue(abs(k.busy - wantBusy) < 1e-3, "忙碌占比：期望 $wantBusy，得到 ${k.busy}")
        assertTrue(abs(k.medianGap - wantGap) < 1e-2, "空档中位：期望 $wantGap，得到 ${k.medianGap}")
        println("  结构 ${k.kind}，忙碌 ${k.busy}，空档中位 ${k.medianGap}s")
    }

    @Test
    fun `评分一致`() {
        val got = Highlight.score(
            Highlight.ralliesGated(hits, amps, cfg), DoubleArray(0), DoubleArray(0), cfg)
        for (i in expected.indices) {
            assertTrue(abs(got[i].score - field(expected[i], "score")) < 1e-4,
                "第 $i 个回合评分：Python ${field(expected[i], "score")}，Kotlin ${got[i].score}")
        }
        println("  评分 ${got.size} 个全部一致")
    }

    @Test
    fun `三个主题的排序一致`() {
        val rs = Highlight.score(
            Highlight.ralliesGated(hits, amps, cfg), DoubleArray(0), DoubleArray(0), cfg)
        for ((theme, key) in listOf("longest" to "rank_longest",
                                    "power" to "rank_power",
                                    "best" to "rank_best")) {
            val want = nums(key)
            val got = Highlight.rank(rs, theme, cfg).take(10).map { it.start }
            assertEquals(want.size, got.size, "$theme 的条数")
            for (i in want.indices) {
                assertTrue(abs(got[i] - want[i]) < 1e-6,
                    "$theme 第 ${i + 1} 名：Python ${want[i]}，Kotlin ${got[i]}")
            }
            println("  $theme 前 ${got.size} 名一致")
        }
    }

    @Test
    fun `最长相持不返回短于三秒的回合`() {
        val rs = Highlight.score(
            Highlight.ralliesGated(hits, amps, cfg), DoubleArray(0), DoubleArray(0), cfg)
        val got = Highlight.rank(rs, "longest", cfg)
        assertTrue(got.all { it.duration >= cfg.longestMinS },
            "有回合短于 ${cfg.longestMinS} 秒：${got.filter { it.duration < cfg.longestMinS }.map { it.duration }}")
        println("  最长相持 ${got.size} 个，最短 ${"%.1f".format(got.minOfOrNull { it.duration } ?: 0.0)} 秒")
    }
}
