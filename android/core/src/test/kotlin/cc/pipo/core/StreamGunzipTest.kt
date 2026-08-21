package cc.pipo.core

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.FilterInputStream
import java.security.MessageDigest
import java.util.zip.GZIPInputStream
import java.util.zip.GZIPOutputStream
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * 「边下边解压、同时算两个摘要」这套读法本身对不对。
 *
 * 为什么单独测：在模拟器上跑真实下载受磁盘空间限制（模型解压后 323 MB，
 * 那台设备只剩 336 MB），失败时很难分清是**代码错**还是**空间不够** ——
 * 实测就绕了很久。把逻辑拿到 JVM 上单独验，环境因素就排除了。
 *
 * 特别要验的是 FilterInputStream 那一层：GZIPInputStream 内部只会调
 * `read(byte[], int, int)`，如果只重写了 `read()` 单字节版本，
 * 计数和摘要就会全是 0，而解压本身照常工作 —— 症状是「进度条不动但下完了」。
 */
class StreamGunzipTest {

    private fun sha(b: ByteArray) = MessageDigest.getInstance("SHA-256")
        .digest(b).joinToString("") { "%02x".format(it) }

    @Test
    fun `流式解压得到原始字节，且压缩流的摘要算不全`() {
        // 造一段可压缩但不平凡的数据，5 MB 足够跨很多次缓冲区
        val raw = ByteArray(5 shl 20)
        val rng = Random(42)
        for (i in raw.indices) raw[i] = (rng.nextInt(64) + i % 7).toByte()

        val gzBytes = ByteArrayOutputStream().also { bos ->
            GZIPOutputStream(bos).use { it.write(raw) }
        }.toByteArray()

        var counted = 0L
        val outMd = MessageDigest.getInstance("SHA-256")
        val out = ByteArrayOutputStream()

        val counting = object : FilterInputStream(ByteArrayInputStream(gzBytes)) {
            override fun read(b: ByteArray, off: Int, len: Int): Int {
                val n = super.read(b, off, len)
                if (n > 0) counted += n
                return n
            }
        }
        GZIPInputStream(counting, 1 shl 16).use { gz ->
            val buf = ByteArray(1 shl 20)
            while (true) {
                val n = gz.read(buf)
                if (n < 0) break
                out.write(buf, 0, n)
                outMd.update(buf, 0, n)
            }
        }

        // 解压本身完全正确
        assertEquals(raw.size, out.size(), "解压后的长度")
        assertContentEquals(raw, out.toByteArray(), "解压后的内容")
        assertEquals(sha(raw), outMd.digest().joinToString("") { "%02x".format(it) },
            "解压后的摘要")

        // **但外层数不全压缩字节。** gzip 尾部的 CRC32 + ISIZE 不经过这层，
        // 稳定少几个字节 —— 所以「顺手对压缩流也算个 sha256」的做法
        // 会让每一次下载都在最后一步被拒。这个测试把这件事钉死，
        // 免得哪天有人觉得「多一层校验更稳」再加回去。
        assertTrue(counted < gzBytes.size,
            "外层应当数不全：counted=$counted，实际 ${gzBytes.size}")
        println("  解压 ${out.size()} 字节正确；外层只数到 $counted / ${gzBytes.size}" +
                "（差 ${gzBytes.size - counted} 字节，是 gzip 尾部）")
    }
}
