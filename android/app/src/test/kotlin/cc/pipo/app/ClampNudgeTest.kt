package cc.pipo.app

import androidx.media3.common.util.UnstableApi
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * 每段头尾 ±0.1 秒微调的夹取规则。
 *
 * 为什么要有这个测试
 * ------------------
 * 这段逻辑原来写在界面的 onNudge 闭包里，**一行测试都没有**，而它要同时
 * 满足四个约束。我当时只在截图里看到按钮渲染出来了，就当它能用 ——
 * 那正是「编译通过 / 界面出来了 ≠ 功能对」的老毛病。
 *
 * 挪成纯函数之后就能在 JVM 上逐条钉死，不需要设备（而设备恰恰经常不在）。
 *
 * 用一组固定的段来测：
 *   段0 [ 1.0,  3.0]
 *   段1 [ 5.0,  8.0]
 *   段2 [10.0, 12.0]
 *   素材总长 20 秒
 */
@UnstableApi
class ClampNudgeTest {

    private val clips = listOf(
        Pipeline.Clip(1.0, 3.0),
        Pipeline.Clip(5.0, 8.0),
        Pipeline.Clip(10.0, 12.0),
    )
    private val dur = 20.0

    private fun nudge(
        i: Int, dh: Double, dt: Double,
        cur: Map<Int, Pipeline.Nudge> = emptyMap(),
    ) = Pipeline.clampNudge(clips, cur, i, dh, dt, dur)

    @Test
    fun 正常范围内原样生效() {
        val n = nudge(1, -0.1, 0.1)
        assertEquals(-0.1, n.head, 1e-9)
        assertEquals(0.1, n.tail, 1e-9)
    }

    @Test
    fun 多次微调会累加() {
        var cur = emptyMap<Int, Pipeline.Nudge>()
        repeat(3) { cur = cur + (1 to nudge(1, -0.1, 0.0, cur)) }
        assertEquals("点三下应当累加到 -0.3", -0.3, cur[1]!!.head, 1e-9)
    }

    /** 段1 的头往前最多到段0 的结尾 3.0，也就是 -2.0。 */
    @Test
    fun 头部不越过前一段的结尾() {
        var cur = emptyMap<Int, Pipeline.Nudge>()
        repeat(50) { cur = cur + (1 to nudge(1, -0.1, 0.0, cur)) }
        assertEquals(-2.0, cur[1]!!.head, 1e-9)
        assertTrue("越过了前一段：起点 ${5.0 + cur[1]!!.head} < 3.0",
            5.0 + cur[1]!!.head >= 3.0 - 1e-9)
    }

    /** 段1 的尾往后最多到段2 的开头 10.0，也就是 +2.0。 */
    @Test
    fun 尾部不越过后一段的开头() {
        var cur = emptyMap<Int, Pipeline.Nudge>()
        repeat(50) { cur = cur + (1 to nudge(1, 0.0, 0.1, cur)) }
        assertEquals(2.0, cur[1]!!.tail, 1e-9)
    }

    /**
     * **这条是代码里原来漏掉的。** 最后一段的上界曾经是 Double.MAX_VALUE，
     * 尾巴能一直拉到素材长度之外 —— 注释里写了「不跑出素材」，代码里没有。
     */
    @Test
    fun 最后一段的尾部不跑出素材() {
        var cur = emptyMap<Int, Pipeline.Nudge>()
        repeat(200) { cur = cur + (2 to nudge(2, 0.0, 0.1, cur)) }
        val end = 12.0 + cur[2]!!.tail
        assertTrue("跑出素材了：结尾 $end > $dur", end <= dur + 1e-9)
        assertEquals(dur, end, 1e-9)
    }

    /** 第一段的头往前最多到 0。 */
    @Test
    fun 第一段的头部不跑到负数() {
        var cur = emptyMap<Int, Pipeline.Nudge>()
        repeat(50) { cur = cur + (0 to nudge(0, -0.1, 0.0, cur)) }
        assertTrue("跑到负数了：起点 ${1.0 + cur[0]!!.head}", 1.0 + cur[0]!!.head >= -1e-9)
    }

    @Test
    fun 一段不会被缩到小于最短长度() {
        var cur = emptyMap<Int, Pipeline.Nudge>()
        // 头往后推、尾往前收，两边一起挤
        repeat(100) { cur = cur + (1 to nudge(1, 0.1, -0.1, cur)) }
        val n = cur[1]!!
        val len = (8.0 + n.tail) - (5.0 + n.head)
        assertTrue("被挤到 $len 秒，短于 ${Pipeline.MIN_CLIP_S}",
            len >= Pipeline.MIN_CLIP_S - 1e-9)
    }

    /**
     * **相邻两段都往中间挤时不能崩。**
     * coerceIn 在 min > max 时抛「Cannot coerce value to an empty range」——
     * 崩在一次微调上，比夹不准糟糕得多。
     */
    @Test
    fun 相邻两段互相挤压时不抛异常() {
        var cur = emptyMap<Int, Pipeline.Nudge>()
        repeat(40) {
            cur = cur + (0 to nudge(0, 0.0, 0.1, cur))      // 段0 尾巴往后
            cur = cur + (1 to nudge(1, -0.1, 0.0, cur))     // 段1 头往前
            cur = cur + (1 to nudge(1, 0.0, 0.1, cur))      // 段1 尾巴往后
            cur = cur + (2 to nudge(2, -0.1, 0.0, cur))     // 段2 头往前
        }
        // 没抛异常就算过；顺带确认三段仍然不重叠
        val ends = clips.mapIndexed { i, c ->
            val n = cur[i] ?: Pipeline.Nudge()
            (c.start + n.head) to (c.end + n.tail)
        }
        for (k in 0 until ends.size - 1) {
            assertTrue("段$k 和段${k + 1} 重叠了：${ends[k]} vs ${ends[k + 1]}",
                ends[k].second <= ends[k + 1].first + 1e-9)
        }
    }

    @Test
    fun 素材时长未知时不按它夹() {
        val n = Pipeline.clampNudge(clips, emptyMap(), 2, 0.0, 5.0, 0.0)
        assertEquals("时长未知就不该被 0 夹住", 5.0, n.tail, 1e-9)
    }
}
