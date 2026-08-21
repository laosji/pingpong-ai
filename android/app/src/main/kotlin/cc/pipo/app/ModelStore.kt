package cc.pipo.app

import android.content.Context
import java.io.File
import java.security.MessageDigest
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

    // **注意这里要的是原始 cnn14.onnx，不是 .gz** —— 续传要求文件可按字节
    // 定位，而 gzip 流不能从中间接上。见 download 的说明。
    private const val BASE = "https://pub-a7dae8fcbe6a4b418f443b1528d3114e.r2.dev/v1"
    private const val LOCAL = "cnn14.onnx"
    /** 下载体积 = 原始模型体积（不再下 .gz，见 download 的说明）。 */
    const val DOWNLOAD_BYTES = 323_011_186L
    private const val FINAL_BYTES = 323_011_186L
    private const val SHA = "22b41aef1908df6612fb9cf18640519d5b1398b7c79d3d46b0f5856470b1e3e0"

    fun file(ctx: Context): File = File(File(ctx.filesDir, "models").apply { mkdirs() }, LOCAL)

    fun ready(ctx: Context): Boolean {
        val f = file(ctx)
        return f.exists() && f.length() >= FINAL_BYTES * 0.99
    }

    /**
     * 清掉没法复用的半截文件。启动时跑一次。
     *
     * **只删 `.un_`，不删 `.dl_`。** 前者是从安装包展开到一半的残留，
     * 重来一次只要几秒，留着没意义；后者是**下载的断点**，删了用户就得
     * 重下 323 MB。这个函数原来两个都删 —— 那是为「半截文件是垃圾」的
     * 旧实现写的，加了续传之后半截文件变成了资产，再删就是帮倒忙。
     *
     * `.dl_` 的安全性由别处保证：续传前查体积不超过总长，
     * 拼完整段校验 sha256，不过就地删掉重来。
     */
    fun cleanup(ctx: Context) {
        File(ctx.filesDir, "models").listFiles()
            ?.filter { it.name.startsWith(".un_") }
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
            "重试会从断点继续，不用重下。")

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

    /**
     * 下载模型。**支持断点续传。**
     *
     * 为什么下原始文件而不是 .gz
     * --------------------------
     * 原来下 .gz 边下边解压，磁盘峰值只有 323 MB（而不是 302+323=625），
     * 代价是**半截文件是解压后的内容，没法续传** —— 断在 99% 也要从头再下
     * 301 MB。实测就撞到过：r2.dev 在 210/301 MB 处卡死，用户白等两分钟。
     *
     * 改成下原始文件：多花 7%（308 vs 288 MB），换来的是
     *   * 断了能从断点续 —— 这是最常见的失败，301 MB 的下载断一次很正常
     *   * 峰值磁盘不变，还是 323 MB
     *   * 代码里一行 gzip 都不用
     *
     * 服务端不支持 Range 时自动退回从头下，不会更差。
     */
    fun download(ctx: Context, onProgress: (Progress) -> Unit) {
        // 包里带了就直接展开，一个字节都不用联网
        if (bundled(ctx)) return unpack(ctx, onProgress)
        checkSpace(ctx)
        val dir = File(ctx.filesDir, "models").apply { mkdirs() }
        val part = File(dir, ".dl_cnn14.part")

        // 上次断在哪。**必须校验整段而不是只校验新下的部分** ——
        // 续传拼出来的文件要么整体对，要么整体不对，分段算摘要没有意义。
        var have = if (part.exists()) part.length() else 0L
        if (have > FINAL_BYTES) { part.delete(); have = 0L }

        try {
            var done = have
            (URL("$BASE/$LOCAL").openConnection() as HttpsURLConnection).apply {
                connectTimeout = 30_000
                readTimeout = 60_000
                setRequestProperty("User-Agent", "Pipo-Android")
                if (have > 0) setRequestProperty("Range", "bytes=$have-")
            }.use { conn ->
                // 206 = 服务端接受了续传；200 = 不支持，从头给
                val resuming = conn.responseCode == 206
                if (!resuming && have > 0) { part.delete(); done = 0L }
                conn.inputStream.use { src ->
                    java.io.FileOutputStream(part, resuming).buffered(1 shl 20).use { out ->
                        val buf = ByteArray(1 shl 20)
                        while (true) {
                            val n = src.read(buf)
                            if (n < 0) break
                            out.write(buf, 0, n)
                            done += n
                            onProgress(Progress.Downloading(done, FINAL_BYTES))
                        }
                    }
                }
            }
            onProgress(Progress.Finishing)
            // 摘要在这里整段算一次。续传意味着数据来自两次连接，
            // 边下边算的摘要跨不了连接。
            val md = MessageDigest.getInstance("SHA-256")
            part.inputStream().buffered(1 shl 20).use { i ->
                val buf = ByteArray(1 shl 20)
                while (true) {
                    val n = i.read(buf)
                    if (n < 0) break
                    md.update(buf, 0, n)
                }
            }
            check(md.digest().hex() == SHA) {
                // 校验不过说明拼出来的东西是坏的，留着只会让下次续传继续错
                part.delete()
                "模型校验不通过 —— 已清掉重下"
            }
            check(part.renameTo(file(ctx))) { "无法写入模型文件" }
        } catch (e: Throwable) {
            // **不删半截文件** —— 那正是下次续传的起点。
            // 只有校验失败才删（在上面），因为那时候文件本身是坏的。
            if (e.message?.contains("ENOSPC") == true) {
                part.delete()
                val free = File(ctx.filesDir.absolutePath).usableSpace / 1_000_000
                throw NotEnoughSpace(FINAL_BYTES / 1_000_000, free)
            }
            if (e is java.net.SocketTimeoutException || e is java.net.UnknownHostException ||
                e is java.net.ConnectException || e is javax.net.ssl.SSLException ||
                (e is java.io.IOException && e !is java.io.FileNotFoundException)) {
                throw Interrupted(if (part.exists()) part.length() else 0L)
            }
            throw e
        }
    }

    private fun ByteArray.hex() = joinToString("") { "%02x".format(it) }

    private inline fun <T> HttpsURLConnection.use(block: (HttpsURLConnection) -> T): T =
        try { block(this) } finally { disconnect() }
}
