package cc.pipo.app

import android.content.Context
import java.io.File
import java.security.MessageDigest
import java.util.zip.GZIPInputStream
import javax.net.ssl.HttpsURLConnection
import java.net.URL

/**
 * 按需下载的声学模型 —— 对应桌面版的 ppai/assets.py。
 *
 * 302 MB 不进安装包。它是**冻结的**预训练骨干，我们不训它、也不会更新，
 * 没有理由让每次发版重新分发一遍；而且 Play 商店对安装包体积有实际约束。
 *
 * **但它不是可选的。** 缺了它重排会原样返回候选（击球检测准确率
 * 从 0.68 掉回 0.38），而且不报错 —— 只让成片悄悄变差。所以策略是
 * **挡住而不是降级**：没有模型就不让开始，明确提示去下载。
 *
 * 两层校验，缺一不可：
 *   * 下载物的 sha256 —— 挡「没下全 / CDN 返回错误页」
 *   * **解压后**的 sha256 —— 挡「解压出来是坏的」
 * 少任何一层，一个能存下来但用不了的模型都会混过去，
 * 而 ONNX 要到加载时才报错，报的还是格式错误，很难联想到是下载的问题。
 */
object ModelStore {

    private const val BASE = "https://pub-a7dae8fcbe6a4b418f443b1528d3114e.r2.dev/v1"
    private const val REMOTE = "cnn14.onnx.gz"
    private const val LOCAL = "cnn14.onnx"
    const val DOWNLOAD_BYTES = 301_692_277L
    private const val FINAL_BYTES = 323_011_186L
    private const val SHA = "22b41aef1908df6612fb9cf18640519d5b1398b7c79d3d46b0f5856470b1e3e0"

    fun file(ctx: Context): File = File(File(ctx.filesDir, "models").apply { mkdirs() }, LOCAL)

    fun ready(ctx: Context): Boolean {
        val f = file(ctx)
        return f.exists() && f.length() >= FINAL_BYTES * 0.99
    }

    /** 清掉上次中断留下的半截文件。启动时跑一次。 */
    fun cleanup(ctx: Context) {
        File(ctx.filesDir, "models").listFiles()
            ?.filter { it.name.startsWith(".dl_") || it.name.startsWith(".un_") }
            ?.forEach { it.delete() }
    }

    sealed interface Progress {
        data class Downloading(val done: Long, val total: Long) : Progress
        /** 从安装包里展开。**和下载分开** —— 打包版根本没联网，
         *  界面上写「下载声学模型」是骗人的。 */
        data class Unpacking(val done: Long, val total: Long) : Progress
        data object Finishing : Progress
    }

    /**
     * 下载被网络打断。**必须有自己的类型** —— 系统抛出来的消息是一个单词
     * 「timeout」，原样显示给用户等于没说。而 301 MB 的下载中途断一次
     * 太常见了，这是最需要说人话的一条。
     */
    class Interrupted(doneBytes: Long) : Exception(
        "下载中断了（已下 ${doneBytes / 1_000_000} / ${DOWNLOAD_BYTES / 1_000_000} MB）。" +
            "连上 WiFi 后重试 —— 会从头下，所以别用流量。")

    class NotEnoughSpace(val needMb: Long, val freeMb: Long) : Exception(
        "存储空间不够：解压后的模型要 ${needMb} MB，这台设备只剩 ${freeMb} MB。" +
            "清理一些空间再试 —— 模型只下这一次，之后不再占额外空间。")

    /**
     * 下载 → **边下边解压** → 两个摘要同时算 → 原子改名。**同步**，调用方放到后台。
     *
     * 为什么不是「先下 .gz，再解压成第二个文件」
     * -----------------------------------------
     * 那样磁盘峰值是 302 + 323 = **625 MB**。手机上这个量级很容易不够 ——
     * 实测模拟器剩 375 MB 时直接失败。改成流式之后峰值只有 323 MB，
     * 而且省掉一次对 300 MB 的额外读取（原来要单独再读一遍算摘要）。
     *
     * 只校验**解压后**的 sha256，不校验压缩流
     * ----------------------------------------
     * 一开始两个都算，结果压缩流那个**永远对不上**：gzip 尾部的
     * CRC32 + ISIZE 这 8 个字节 GZIPInputStream 不经过外层包装读，
     * 计数会稳定少 4 字节（StreamGunzipTest 固化了这个事实）。
     * 那条检查会让每一次下载都在最后一步被拒 —— 而模拟器上是先撞 ENOSPC，
     * 差点就把它当成「空间不够」放过去了。
     *
     * 而且它本来就是多余的：解压后的摘要已经涵盖了它。下载截断 → gzip
     * 直接抛异常或输出对不上；下载损坏 → CRC 校验失败或输出对不上。
     */
    /**
     * 先查空间再下载。**不查的话会下满 300 MB 才在写盘时炸 ENOSPC** ——
     * 实测就是这样：用户等了两分钟，最后看到一句系统原文的
     * 「write failed: ENOSPC」，既不知道发生了什么也不知道该做什么。
     *
     * 留 8% 余量：文件系统本身要开销，而且下载期间用户可能在拍新视频。
     */
    private fun checkSpace(ctx: Context) {
        val free = File(ctx.filesDir.absolutePath).usableSpace
        val need = (FINAL_BYTES * 1.08).toLong()
        if (free < need) throw NotEnoughSpace(need / 1_000_000, free / 1_000_000)
    }

    /**
     * 安装包里带没带模型。带了就不用下载。
     *
     * 用 `-PpipoBundleModel` 构建的版本会把原始模型放进 assets，
     * APK 自己的 deflate 压到 288 MB，成包 308 MB。
     * 那种包**上不了商店**（APK 上限 100 MB、AAB 基础模块 150 MB），
     * 只用于直接发给某个人试用。商店版没有这个 asset，走下载。
     */
    private fun bundled(ctx: Context): Boolean =
        runCatching { ctx.assets.open(LOCAL).close(); true }.getOrDefault(false)

    /**
     * 把打包进来的模型展开到 filesDir。
     *
     * 为什么要复制一份而不是直接从 assets 读：ONNX Runtime 要一个真实的
     * 文件路径才能 mmap 权重。从 assets 读只能整个读进内存，
     * 323 MB 进堆，中端机会被系统直接杀掉。
     *
     * **仍然校验 sha256。** 它就在自己的安装包里，看着不可能坏 ——
     * 但解压到一半没空间、或者写盘时被系统杀掉，都会留下一个能存下来
     * 却用不了的文件，而 ONNX 要到加载时才报错，报的还是格式错误。
     * 校验一次几秒钟，换的是「出问题时知道是出了什么问题」。
     */
    private fun unpack(ctx: Context, onProgress: (Progress) -> Unit) {
        checkSpace(ctx)
        val dir = File(ctx.filesDir, "models").apply { mkdirs() }
        val part = File(dir, ".un_cnn14.part")
        val md = MessageDigest.getInstance("SHA-256")
        try {
            var done = 0L
            // assets 里放的是**原始模型**（见 build.gradle 的说明），
            // APK 的 deflate 由 AssetManager 透明解开，这里不需要 gzip。
            ctx.assets.open(LOCAL).use { src ->
                part.outputStream().buffered(1 shl 20).use { out ->
                    val buf = ByteArray(1 shl 20)
                    while (true) {
                        val n = src.read(buf)
                        if (n < 0) break
                        out.write(buf, 0, n)
                        md.update(buf, 0, n)
                        done += n
                        onProgress(Progress.Unpacking(done, FINAL_BYTES))
                    }
                }
            }
            onProgress(Progress.Finishing)
            check(md.digest().hex() == SHA) { "随包模型校验不通过 —— 安装包可能不完整" }
            check(part.renameTo(file(ctx))) { "无法写入模型文件" }
        } catch (e: Throwable) {
            part.delete()
            if (e.message?.contains("ENOSPC") == true) {
                val free = File(ctx.filesDir.absolutePath).usableSpace / 1_000_000
                throw NotEnoughSpace(FINAL_BYTES / 1_000_000, free)
            }
            throw e
        }
    }

    /** 需要联网吗。界面靠它决定要不要提示「第一次要下 301 MB」。 */
    fun needsNetwork(ctx: Context): Boolean = !ready(ctx) && !bundled(ctx)

    fun download(ctx: Context, onProgress: (Progress) -> Unit) {
        // 包里带了就直接解压，一个字节都不用联网
        if (bundled(ctx)) return unpack(ctx, onProgress)
        checkSpace(ctx)
        val dir = File(ctx.filesDir, "models").apply { mkdirs() }
        var done = 0L
        val part = File(dir, ".dl_cnn14.part")
        val outMd = MessageDigest.getInstance("SHA-256")
        try {
            (URL("$BASE/$REMOTE").openConnection() as HttpsURLConnection).apply {
                connectTimeout = 30_000
                readTimeout = 60_000
                setRequestProperty("User-Agent", "Pipo-Android")
            }.use { conn ->
                val total = conn.contentLengthLong.takeIf { it > 0 } ?: DOWNLOAD_BYTES
                // 提到外层：catch 里要用它告诉用户下到哪了
                done = 0L
                // 计数包在网络流上，进度按**压缩字节**算 —— 那才是用户在等的东西。
                // 注意 GZIPInputStream 走的是数组版 read，单字节版基本不会被调到。
                val counting = object : java.io.FilterInputStream(conn.inputStream) {
                    override fun read(b: ByteArray, off: Int, len: Int): Int {
                        val n = super.read(b, off, len)
                        if (n > 0) {
                            done += n
                            onProgress(Progress.Downloading(done, total))
                        }
                        return n
                    }
                }
                GZIPInputStream(counting, 1 shl 16).use { gz ->
                    part.outputStream().buffered(1 shl 20).use { out ->
                        val buf = ByteArray(1 shl 20)
                        while (true) {
                            val n = gz.read(buf)
                            if (n < 0) break
                            out.write(buf, 0, n)
                            outMd.update(buf, 0, n)
                        }
                    }
                }
            }
            onProgress(Progress.Finishing)
            check(outMd.digest().hex() == SHA) {
                "模型校验不通过 —— 多半是没下全或下到了错的东西，重试一次"
            }
            check(part.length() >= FINAL_BYTES * 0.99) { "解压出来的大小不对" }
            check(part.renameTo(file(ctx))) { "无法写入模型文件" }
        } catch (e: Throwable) {
            part.delete()
            // 下载途中也可能被别的应用把空间吃掉。把系统的 ENOSPC 换成
            // 用户能照着做的话 —— 原文「write failed: ENOSPC」谁也不知道该干嘛。
            if (e.message?.contains("ENOSPC") == true || e is java.io.IOException &&
                e.message?.contains("No space") == true) {
                val free = File(ctx.filesDir.absolutePath).usableSpace / 1_000_000
                throw NotEnoughSpace(FINAL_BYTES / 1_000_000, free)
            }
            // 网络中断。**这条一定要翻译** —— 实测 release 包在 210/301 MB
            // 处卡住后抛出来的消息就是一个单词「timeout」，界面上原样显示，
            // 用户既不知道是什么超时、也不知道该干什么。
            // 而且这是最常见的失败：301 MB 的下载，中途断一次很正常。
            if (e is java.net.SocketTimeoutException || e is java.net.UnknownHostException ||
                e is java.net.ConnectException || e is javax.net.ssl.SSLException ||
                (e is java.io.IOException && e !is java.io.FileNotFoundException)) {
                throw Interrupted(done)
            }
            throw e
        }
    }

    private fun ByteArray.hex() = joinToString("") { "%02x".format(it) }

    private inline fun <T> HttpsURLConnection.use(block: (HttpsURLConnection) -> T): T =
        try { block(this) } finally { disconnect() }
}
