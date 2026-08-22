package cc.pipo.app

import android.content.Context
import android.media.MediaCodecList
import android.os.Build
import java.io.File
import java.net.URL
import javax.net.ssl.HttpsURLConnection

/**
 * 出问题时用户能一键复制、发给我们的信息。
 *
 * 为什么必须有这个
 * ----------------
 * 一台华为机器上卡在「拼接成片」，而我们**什么都拿不到** —— 用户不会用
 * adb，应用对她来说是个黑盒，对我们也是。没有这个文件的话，
 * 唯一的排查手段是「你能再试一次吗」。
 *
 * 收什么、不收什么
 * ----------------
 * 收：机型、系统版本、架构、这台机器有哪些视频编码器、各阶段耗时、
 *     异常类型和消息、剩余空间。
 * **不收**：文件名、路径、相册内容、任何能定位到人的东西。
 * 录像本身更是一帧都不碰 —— 界面上写着「录像不会离开这台手机」，
 * 诊断信息不能让这句话变成谎话。
 *
 * 编码器列表是这里最关键的一项：卡在切片时，最可能的原因就是
 * 某个厂商的编码器和 Media3 的组合有问题，而不同厂商的编码器
 * 名字完全不同（高通 OMX.qcom.*、海思 OMX.hisi.*、联发科 OMX.MTK.*）。
 */
object Diagnostics {

    /** 各阶段耗时。出问题时「卡在哪一步、前面几步花了多久」是第一手线索。 */
    private val marks = LinkedHashMap<String, Long>()
    private var t0 = 0L
    /** 这次处理的素材尺寸。**首要嫌疑就在这**，见 encoders() 的注释。 */
    private var source: Pair<Int, Int>? = null

    fun setSource(w: Int, h: Int) { source = w to h }

    /**
     * 正在跑分析/剪辑。只有这段时间里退到后台才值得记 —— 见 [onBackground]。
     */
    @Volatile
    var busy = false
    private var bg = 0

    /**
     * 应用退到后台了。
     *
     * **这是「卡在剪辑」的头号嫌疑。** 整条分析加剪辑跑在 Activity 的协程里，
     * 没有前台服务也没有唤醒锁 —— 一旦退到后台，系统随时可以冻结或者直接
     * 杀掉这个进程，而 EMUI 在这件事上比 AOSP 激进得多。用户回来看到的
     * 就是一个永远不动的进度条。
     *
     * 现在处理期间会保持屏幕常亮（见 MainActivity），息屏这条堵住了；
     * 但用户主动切走仍然会走到这里，所以记进时间线 —— 下次拿到报告时，
     * 「第几秒退到后台、之后再没有新阶段」是一眼能看出来的形状。
     */
    fun onBackground() {
        if (!busy) return
        bg++
        mark("退到后台#$bg")
    }

    fun reset() {
        marks.clear()
        bg = 0
        t0 = System.currentTimeMillis()
    }

    fun mark(name: String) {
        if (t0 == 0L) reset()
        marks[name] = System.currentTimeMillis() - t0
    }

    /**
     * 这台机器能用的视频编码器。
     *
     * 只列**编码器**，不列解码器 —— 解码到处都能跑，出问题的一直是编码。
     */
    private fun encoders(): String {
        val out = ArrayList<String>()
        runCatching {
            MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos.forEach { c ->
                if (c.isEncoder && c.supportedTypes.any { it.equals("video/avc", true) }) {
                    // 硬件还是软件：软编在老机器上可能慢到看着像卡死
                    val hw = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q)
                        (if (c.isHardwareAccelerated) "硬件" else "软件") else "?"
                    // **对齐要求和最大分辨率是首要嫌疑。**
                    // pickCanvas 只保证偶数，而厂商编码器（尤其海思）常要求
                    // 16 的倍数，不满足时表现是挂起而不是报错。
                    // 手机最常见的 1080p 竖屏，宽度 1080 % 16 = 8 —— 不对齐。
                    // 而我们测过的素材（480x848、1024x512、544x960）
                    // 恰好全部 16 对齐，所以从没触发过。
                    val v = runCatching {
                        c.getCapabilitiesForType("video/avc").videoCapabilities
                    }.getOrNull()
                    val info = if (v == null) "" else
                        " 对齐${v.widthAlignment}x${v.heightAlignment}" +
                            " 上限${v.supportedWidths.upper}x${v.supportedHeights.upper}"
                    out.add("${c.name}($hw$info)")
                }
            }
        }
        return if (out.isEmpty()) "一个 H.264 编码器都没找到" else out.joinToString(", ")
    }

    fun report(ctx: Context, err: Throwable?): String = buildString {
        appendLine("Pipo 诊断信息")
        appendLine("机型 ${Build.MANUFACTURER} ${Build.MODEL}")
        appendLine("系统 Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})")
        appendLine("架构 ${Build.SUPPORTED_ABIS.joinToString(",")}")
        appendLine("版本 ${BuildConfig.VERSION_NAME} (${BuildConfig.BUILD_TYPE})")
        appendLine("剩余空间 ${File(ctx.filesDir.absolutePath).usableSpace / 1_000_000} MB")
        val rt = Runtime.getRuntime()
        appendLine("堆 ${(rt.totalMemory() - rt.freeMemory()) / 1_000_000}"
            + " / ${rt.maxMemory() / 1_000_000} MB")
        appendLine("模型 ${if (ModelStore.ready(ctx)) "就位" else "未就位"}")
        appendLine("H.264 编码器 ${encoders()}")
        source?.let {
            appendLine("素材 ${it.first}x${it.second}"
                + "（宽%16=${it.first % 16} 高%16=${it.second % 16}"
                + (if (it.first % 16 == 0 && it.second % 16 == 0) "，对齐）" else "，**不对齐**）"))
        }
        if (marks.isNotEmpty()) {
            appendLine("各阶段（毫秒）")
            marks.forEach { (k, v) -> appendLine("  $k  $v") }
        }
        if (err != null) {
            appendLine("异常 ${err.javaClass.name}")
            appendLine("消息 ${err.message}")
            // 只留前几帧：够定位，又不会长到用户不愿意发
            err.stackTrace.take(6).forEach { appendLine("  at $it") }
        }
    }

    /** 收集端。和反馈回流是同一个 Worker，只是另一条路由。 */
    private const val ENDPOINT =
        "https://pipo-feedback.laosji.workers.dev/diag"

    /**
     * 出错时自动传一份回来。**只在 BuildConfig.AUTO_DIAG 打开时**，
     * 那是给自己人测试的包用的，正式版必须先问过用户。
     *
     * 三条硬约束：
     *  * **绝不阻塞** —— 单开线程，失败就算了。诊断是附加品，
     *    不能让「传不上去」变成用户看到的第二个错误。
     *  * **绝不重试** —— 传丢一份无所谓，而重试会在没网的地方变成
     *    一串后台请求，耗电又没意义。
     *  * **内容和界面上那个「复制」按钮完全一样** —— 不存在
     *    「传回去的比给用户看的多」这种事。
     */
    fun autoSend(ctx: Context, err: Throwable?) {
        if (!BuildConfig.AUTO_DIAG) return
        val body = report(ctx, err)
        Thread {
            runCatching {
                (URL(ENDPOINT).openConnection() as HttpsURLConnection).apply {
                    requestMethod = "POST"
                    doOutput = true
                    connectTimeout = 10_000
                    readTimeout = 10_000
                    setRequestProperty("Content-Type", "text/plain; charset=utf-8")
                    setRequestProperty("X-Pipo-Install", installId(ctx))
                }.use { c ->
                    c.outputStream.use { it.write(body.toByteArray()) }
                    c.responseCode
                }
            }
        }.apply { isDaemon = true }.start()
    }

    /**
     * 安装编号。**随机生成、只存在本机、和任何账号无关** ——
     * 它只用来把同一台设备的多次上报归到一起，不做别的。
     */
    private fun installId(ctx: Context): String {
        val f = File(ctx.filesDir, "install_id")
        if (f.exists()) return f.readText().trim().take(64)
        val id = java.util.UUID.randomUUID().toString().replace("-", "").take(16)
        runCatching { f.writeText(id) }
        return id
    }

    private inline fun <T> HttpsURLConnection.use(block: (HttpsURLConnection) -> T): T =
        try { block(this) } finally { disconnect() }
}
