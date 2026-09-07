package cc.pipo.cutter

import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * 「用户放弃」必须是一个**类型**，不能是一句话。
 *
 * 这个测试为一个真实的回归而写：取消原本用 `Failed("已取消", null)` 表示，
 * 上层靠 `e.message == "已取消"` 来认。后来给失败消息包了一层人话
 * （「没能把片段拼成成片（…）。点上面的「复制诊断信息」…」），
 * 字符串比较当场失配 —— **用户点了「放弃」，界面弹出一个红色错误卡片。**
 *
 * 改文案不该改变程序状态。所以这里钉的不是文案，而是
 * **Cancelled 和 Failed 是两个不相交的类型**：
 * 任何人再想用「消息里带某几个字」来表达状态，都会在这里绊一跤。
 */
class CancelIsNotFailureTest {

    @Test
    fun 取消不是失败的一种() {
        val c: Cutter.Outcome = Cutter.Outcome.Cancelled
        assertTrue("取消被归成了失败 —— 上层会把它当错误弹给用户",
            c !is Cutter.Outcome.Failed)
        assertTrue("取消被归成了成功 —— 那会拿一个不存在的成片去预览",
            c !is Cutter.Outcome.Ok)
    }

    /**
     * **不允许再用文案表达取消。** 就算有人把 Failed 的消息写成「已取消」，
     * 它也仍然是 Failed，不该被任何地方当成用户放弃。
     */
    @Test
    fun 消息里写着已取消的失败仍然是失败() {
        val f: Cutter.Outcome = Cutter.Outcome.Failed("已取消", null)
        assertTrue("用文案冒充了取消状态", f !is Cutter.Outcome.Cancelled)
    }

    @Test
    fun 三种结局互不重叠() {
        val all = listOf<Cutter.Outcome>(
            Cutter.Outcome.Ok(java.io.File("x"), 1),
            Cutter.Outcome.Failed("boom", null),
            Cutter.Outcome.Cancelled,
        )
        // 每个结局只能落进一个分支。when 是穷尽的，漏一个编译期就会报，
        // 这里再确认运行期也不会有一个值同时命中两条。
        val hits = all.map { o ->
            listOf(
                o is Cutter.Outcome.Ok,
                o is Cutter.Outcome.Failed,
                o is Cutter.Outcome.Cancelled,
            ).count { it }
        }
        assertTrue("有结局同时属于多个分支：$hits", hits.all { it == 1 })
    }
}
