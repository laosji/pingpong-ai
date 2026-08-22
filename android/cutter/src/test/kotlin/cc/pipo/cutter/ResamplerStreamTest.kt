package cc.pipo.cutter

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.random.Random

/**
 * 流式重采样必须和整段重采样**逐位相同**。
 *
 * 这条不成立的话，改动就等于悄悄换掉了检测算法 —— 检测靠谱通量，
 * 重采样的滤波特性一变，检出的击球时刻就跟着变，前面对着 Python
 * 逐点对齐的那些工作全部作废。所以这里不比「接近」，比 `==`。
 *
 * 不规则分块是重点：真实解码器吐出来的块大小本来就不齐，
 * 而分块边界正是流式实现最容易错的地方（历史留少了就少算抽头，
 * 留多了就重复累加）。
 */
class ResamplerStreamTest {

    private fun oneShot(x: FloatArray, inRate: Int, outRate: Int) =
        Resampler.resample(x, inRate, outRate)

    private fun streamed(
        x: FloatArray, inRate: Int, outRate: Int, blocks: List<Int>,
    ): FloatArray {
        val s = Resampler.Stream(inRate, outRate)
        val out = ArrayList<Float>()
        var i = 0
        var bi = 0
        while (i < x.size) {
            val n = minOf(blocks[bi % blocks.size], x.size - i)
            s.feed(x.copyOfRange(i, i + n), n) { out.add(it) }
            i += n; bi++
        }
        s.finish { out.add(it) }
        return FloatArray(out.size) { out[it] }
    }

    private fun check(inRate: Int, outRate: Int, n: Int, blocks: List<Int>) {
        val r = Random(12345)
        val x = FloatArray(n) { r.nextFloat() * 2f - 1f }
        val a = oneShot(x, inRate, outRate)
        val b = streamed(x, inRate, outRate, blocks)
        assertEquals("$inRate->$outRate 输出长度", a.size, b.size)
        var worstAt = -1
        for (i in a.indices) {
            if (a[i] != b[i]) { worstAt = i; break }
        }
        assertTrue(
            "$inRate->$outRate 第 $worstAt 个点不一致：" +
                "整段 ${if (worstAt >= 0) a[worstAt] else 0f} vs " +
                "流式 ${if (worstAt >= 0) b[worstAt] else 0f}",
            worstAt < 0)
    }

    @Test
    fun `48k到16k 整数抽取`() {
        check(48000, 16000, 200_000, listOf(1024, 4096, 777, 1, 16384))
    }

    @Test
    fun `48k到32k 有理比`() {
        check(48000, 32000, 200_000, listOf(1024, 4096, 777, 1, 16384))
    }

    @Test
    fun `44_1k到16k 大质数比`() {
        check(44100, 16000, 120_000, listOf(2048, 333, 9999))
    }

    /** 每次只喂一个采样 —— 最刁钻的分块方式。 */
    @Test
    fun `逐采样喂也一致`() {
        check(48000, 16000, 5_000, listOf(1))
    }

    /** 一次全喂进去，等价于整段。 */
    @Test
    fun `单块等价于整段`() {
        check(48000, 16000, 50_000, listOf(Int.MAX_VALUE))
    }

    /** 比滤波器还短的输入 —— 边界全靠补零，最容易错。 */
    @Test
    fun `极短输入也一致`() {
        check(48000, 16000, 7, listOf(1, 3))
        check(48000, 32000, 40, listOf(7))
    }
}
