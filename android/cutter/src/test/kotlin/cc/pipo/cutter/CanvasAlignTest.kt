package cc.pipo.cutter

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * 输出画布必须 16 对齐、且不超过编码器普遍能接受的分辨率。
 *
 * **这个测试是补一个覆盖空洞。** 一台华为机器卡在「拼接成片」之后才发现：
 * 我们测过的每一段素材（480x848、1024x512、544x960）都**恰好**是 16 的倍数，
 * 而手机最常见的 1080p 竖屏，宽度 1080 % 16 = 8 —— 不对齐。厂商编码器
 * 对齐不满足时常常是**挂起而不是报错**，所以这条路径既没被测到、
 * 出问题时也不会自己喊出来。
 */
class CanvasAlignTest {

    private fun pick(w: Int, h: Int) = Cutter.pickCanvas(listOf(Cutter.Canvas(w, h)))

    @Test
    fun `常见手机分辨率都对齐到 16`() {
        for ((w, h) in listOf(
            1080 to 1920,   // 最常见的竖屏，宽度不是 16 的倍数
            1920 to 1080,
            720 to 1280,
            1440 to 2560,
            544 to 960,     // 我们的原片
            480 to 848,     // 我们的测试片
        )) {
            val c = pick(w, h)
            assertEquals(0, c.width % 16, "宽 ${c.width} 不是 16 的倍数（源 ${w}x$h）")
            assertEquals(0, c.height % 16, "高 ${c.height} 不是 16 的倍数（源 ${w}x$h）")
            println("  ${w}x$h -> ${c.width}x${c.height}")
        }
    }

    @Test
    fun `超高分辨率被压到 1920 以内且比例不跑偏`() {
        for ((w, h) in listOf(3840 to 2160, 2160 to 3840, 7680 to 4320)) {
            val c = pick(w, h)
            assertTrue(maxOf(c.width, c.height) <= 1920,
                "${w}x$h -> ${c.width}x${c.height}，长边超过 1920")
            val src = w.toDouble() / h
            val dst = c.width.toDouble() / c.height
            assertTrue(kotlin.math.abs(src - dst) < 0.05,
                "比例变了：源 %.3f，输出 %.3f".format(src, dst))
            println("  ${w}x$h -> ${c.width}x${c.height}")
        }
    }

    @Test
    fun `极小尺寸不会退化成零`() {
        val c = pick(8, 8)
        assertTrue(c.width >= 16 && c.height >= 16, "退化成了 ${c.width}x${c.height}")
    }
}
