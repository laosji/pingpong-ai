package cc.pipo.app

import android.content.ContentValues
import android.content.Context
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import androidx.media3.common.util.UnstableApi
import cc.pipo.core.Audio
import cc.pipo.core.Highlight
import cc.pipo.core.Rerank
import cc.pipo.cutter.AudioDecode
import cc.pipo.cutter.Cutter
import java.io.File

/**
 * 把各层串成一条流水线。每一层都已经对着 Python 逐点比对过
 * （core 的 16 个 JVM 测试 + cutter 的 7 个仪器测试），这里只负责接线。
 *
 * 顺序和桌面版完全一致：
 *   解码 → 检测 → 重排 → 分组 → 排序 → 切片
 *
 * 主题只有三个 —— 「精彩瞬间」和「失误合集」是**测过之后**去掉的
 * （前者与另两个主题的第一名完全重合，后者的指标只有 48% 命中率），
 * 详见桌面版 README 的「已下线」几节。
 */
@UnstableApi
object Pipeline {

    data class Theme(val id: String, val name: String, val desc: String)

    val THEMES = listOf(
        Theme("auto", "自动", "去掉捡球和等待，一个球都不漏"),
        // 「最可靠」是我们内部的评测结论，不是用户要读的东西 ——
        // 用户看到会反问「那别的不可靠吗」。文案只说这个主题挑什么。
        Theme("power", "扣杀瞬间", "击球声最响的那几个回合"),
        Theme("longest", "最长相持", "来回拍数最多，且不短于 3 秒"),
        Theme("best", "训练集锦", "综合评分最高的若干回合"),
    )

    /**
     * 进度。**分「现在在做」和「刚做完」两类** —— 后者带着数字，
     * 界面把它们攒成一条清单，让两三分钟的等待看得见在干什么，
     * 而不是一个转了很久的圈。数字都是真的，不是编来撑场面的。
     */
    sealed interface Stage {
        /**
         * 诊断时间线里用的名字。
         *
         * **必须是写死的字符串，不能用 `javaClass.simpleName`。** 原来就是
         * 那么写的，debug 包里一切正常，而 release 包被 R8 改名之后，
         * 传回来的报告是这样的：
         *
         *     各阶段（毫秒）
         *       w0  14
         *       v0  201
         *       B0  2722
         *
         * 「卡在哪一步」正是这份报告最核心的一列，混淆之后完全读不出来 ——
         * 而这个错误只会出现在正式包里，也就是只会出现在真实用户身上。
         * 是把第一份真上传的报告取回来读了才发现的。
         */
        val mark: String

        data object Decoding : Stage { override val mark = "解码音频" }
        data class Decoded(val seconds: Double) : Stage { override val mark = "音频就绪" }
        data object Detecting : Stage { override val mark = "检测击球" }
        data class Detected(val n: Int) : Stage { override val mark = "击球检出" }
        /** [percent] 是真百分比：窗数在开始前就算得出来。 */
        data class Reranking(val percent: Int) : Stage { override val mark = "重排中" }
        data class Reranked(val kept: Int, val total: Int) : Stage {
            override val mark = "重排完成"
        }
        data class Grouped(val rallies: Int, val clips: Int) : Stage {
            override val mark = "分组完成"
        }
        data class Cutting(val percent: Int) : Stage { override val mark = "拼接成片" }
    }

    /**
     * 成片里的一段。**这是给界面看的类型**，不用 Highlight.Rally ——
     * Rally 带着一堆检测中间量（power / peakPower / endedWithBounce…），
     * 界面一个都不需要，透出去只会让「删掉一段再重剪」被迫依赖检测细节。
     */
    data class Clip(val start: Double, val end: Double) {
        val duration: Double get() = end - start
    }

    /**
     * [clips] 是成片实际用到的段，**按顺序**，界面靠它做逐段删除。
     * 删完调 [cut] 重来一次即可 —— 不必重新检测。
     */
    data class Result(val file: File, val seconds: Double, val clips: List<Clip>)

    /**
     * 剪不出东西的三种原因，**必须分开报**。
     *
     * 三条路径失败时用户该做的事完全不同：换个安静点的环境重录 / 换素材 /
     * 换个主题。以前它们共用一句「没检测到有效回合」，实测踩到过 ——
     * 门禁判定「这不是乒乓球」时报的却是「没有回合」，
     * 会把人引去检查录音，而真正的问题是选错了素材。
     */
    class NoHits : Exception(
        "几乎没听到击球声。确认这段录像有声音，机位离球台别太远。")

    class NotPingpong : Exception(
        "这段录像看起来不是乒乓球。检测靠击球声，请选一段真正的对打录像。")

    class NoRallies(theme: String) : Exception(
        "识别到了击球，但凑不出「%s」要的回合。换个主题试试。".format(theme))

    /**
     * 跑完整条链路：先分析出要哪几段，再切出来。
     *
     * **拆成两步是为了「删掉一段」不用重新分析。** 分析要解两遍音频、
     * 跑一遍 2048 维嵌入，实测占整条链路的绝大部分时间（45 秒素材上
     * 分析约 150 秒、切片约 20 秒）；而删一段只改切片的输入。
     * 合在一起写的话，用户每删一段就要重等一遍分析。
     */
    fun run(
        ctx: Context, uri: Uri, themeId: String, top: Int = Highlight.DEFAULT_TOP_N,
        onStage: (Stage) -> Unit,
    ): Result = cut(ctx, uri, analyze(ctx, uri, themeId, top, onStage), onStage)

    /** 被用户放弃。和真正的失败分开 —— 界面不该把它当成错误弹出来。 */
    class Cancelled : Exception("已取消")

    /**
     * 分析：解码 → 检测 → 重排 → 按主题挑段。**同步**，调用方放到后台线程。
     *
     * 内存是这里最现实的约束：一小时 48kHz 的素材解到 32kHz 是 460MB float，
     * 中端机会被系统直接杀掉。所以先查时长，超了就明确拒绝，
     * 而不是跑到一半 OOM —— 后者用户看到的是「闪退」，查都没法查。
     */
    fun analyze(
        ctx: Context, uri: Uri, themeId: String, top: Int = Highlight.DEFAULT_TOP_N,
        onStage: (Stage) -> Unit,
    ): List<Clip> {
        val model = ModelStore.file(ctx)
        require(model.exists()) { "声学模型还没下载" }

        val durationS = durationOf(ctx, uri)
        // **必须先挡住时长为 0。** 拿不到时长元数据时 durationOf 返回 0，
        // 而 0 能通过下面的上限检查，于是整条链路照跑两分钟，
        // 最后 select 把每一段都夹到 0 长度、全部丢掉，抛出来的却是
        // 「凑不出这个主题要的回合，换个主题试试」—— 换到天亮也没用。
        require(durationS > 0.0) {
            "读不出这段录像的时长，可能是文件损坏或者格式不支持。换一段试试。"
        }
        require(durationS <= Rerank.MAX_SAFE_SECONDS) {
            "这段录像 %.0f 分钟，超过本机一次能处理的 %d 分钟。".format(
                durationS / 60, Rerank.MAX_SAFE_SECONDS / 60) +
                "先分段再剪 —— 不是不想支持，是整段解码要占几百 MB 内存，硬跑会被系统杀掉。"
        }

        onStage(Stage.Decoding)
        val pcm16 = AudioDecode.decode(ctx, uri, 16000)
        onStage(Stage.Decoded(durationS))

        onStage(Stage.Detecting)
        val det = Audio.detectHits(pcm16, Audio.Config())
        if (det.hits.isEmpty()) throw NoHits()
        onStage(Stage.Detected(det.hits.size))

        onStage(Stage.Reranking(0))
        val pcm32 = AudioDecode.decode(ctx, uri, 32000)
        val kept: DoubleArray
        Rerank.load(model.absolutePath, weightsPath(ctx)).use { m ->
            val (times, feats) = Rerank.embed(pcm32, m) { done, total ->
                onStage(Stage.Reranking(if (total > 0) 100 * done / total else 0))
            }
            // 全片平均概率是「这是不是乒乓球录像」的判据：
            // 实测阴性 0.009-0.010、真实素材 0.379-0.672，差 40 倍。
            val mean = Rerank.meanProbability(feats, m)
            if (mean < 0.10) throw NotPingpong()
            kept = Rerank.apply(det.hits, times, feats, m, 0.6)
        }
        onStage(Stage.Reranked(kept.size, det.hits.size))

        val amps = Audio.hitAmplitudes(kept, det.env, det.frameRate)
        val cfg = Highlight.Config()
        var rs = Highlight.ralliesGated(kept, amps, cfg)
        rs = Highlight.score(rs, DoubleArray(0), DoubleArray(0), cfg)
        // **必须走 select。** 它做的是滤短 + 扩边 + 合并 + 夹到片长 ——
        // 少了这一步，切点会落在「击球声响起的那一帧」，而声音响起时挥拍
        // 已经做完了，观众看到的是球已经飞出去的画面；同一段素材还会多出
        // 一半 0.5-0.7 秒的碎片（实测 8 段里 4 段短于 1.5 秒）。
        val ordered = when (themeId) {
            // 「自动」= 完整版：保留所有回合，只去掉中间的等待
            "auto" -> rs.sortedBy { it.start }
            else -> Highlight.rank(rs, themeId, cfg)
        }
        // 「自动」不设上限：它的卖点就是一个球都不漏
        val n = if (themeId == "auto") ordered.size else top
        val picked = Highlight.select(ordered, durationS, cfg, topN = n)
        if (picked.isEmpty()) {
            throw NoRallies(THEMES.firstOrNull { it.id == themeId }?.name ?: themeId)
        }
        onStage(Stage.Grouped(rs.size, picked.size))
        return picked.map { Clip(it.start, it.end) }
    }

    /**
     * 切片：把选中的段拼成一支成片。**只碰视频，不做任何分析** ——
     * 用户在成片页删掉一段之后重调的就是它。
     *
     * 每次都写一个新文件名而不是覆盖：播放器可能还持着上一份的句柄，
     * 覆盖会让预览停在一个已经变了的文件上。旧的由 [clearOutputs] 收。
     */
    fun cut(
        ctx: Context, uri: Uri, clips: List<Clip>, onStage: (Stage) -> Unit,
        /** 见 Cutter.cut 的说明：阻塞等待看不到协程取消，得由调用方告诉它。 */
        shouldStop: (() -> Boolean)? = null,
    ): Result {
        require(clips.isNotEmpty()) { "一段都不留就没得剪了。至少留一段。" }
        val size = sizeOf(ctx, uri)
        val out = File(outDir(ctx), "pipo_%d.mp4".format(System.currentTimeMillis()))
        val res = Cutter.cut(
            ctx,
            clips.map { Cutter.Segment(uri, it.start, it.end) },
            out, Cutter.pickCanvas(listOf(size)),
            onProgress = { onStage(Stage.Cutting(it)) },
            shouldStop = shouldStop,
        )
        return when (res) {
            is Cutter.Outcome.Ok -> {
                // Transformer 报成功不等于文件还在。实测过一次：磁盘快满时
                // 系统在切完之后、用户点保存之前把文件删了，界面照样显示
                // 「剪好了 8 段 · 0:13　0.0 MB」，点保存才炸出 ENOENT。
                // 空文件是失败，必须在这里就拦住。
                if (res.file.length() <= 0L) throw OutputGone()
                // 用**成片实际时长**，不是各段请求时长之和。切点要对齐帧，
                // 两者会差零点几秒 —— 界面写「0:13」而播放器显示「00:14」，
                // 用户会怀疑是不是漏了一段。
                Result(res.file, res.durationMs / 1000.0, clips)
            }
            is Cutter.Outcome.Failed -> throw RuntimeException(res.message)
        }
    }

    class OutputGone : Exception(
        "成片没能保住，多半是手机存储不够。清点空间再试一次。")

    /**
     * 成片的落点。**不能用 cacheDir** —— 系统在低存储时会直接清空它，
     * 而剪辑恰好常发生在手机快满的时候。实测在剩 30 MB 的机器上，
     * DeviceStorageMonitorService 在切完到点保存之间就把成片删了。
     *
     * filesDir 系统不碰，代价是得自己收拾：每次开新任务前清掉上一次的，
     * 否则剪十次就留十份成片。
     */
    private fun outDir(ctx: Context): File =
        File(ctx.filesDir, "out").apply { mkdirs() }

    /**
     * 清掉留下的成片。用户已经存进相册的那份在 MediaStore 里，不受影响。
     *
     * [keep] 是当前正在预览的那份 —— 用户每删一段就会重切一次，
     * 不留这个例外的话会把播放器正在放的文件删掉。
     */
    fun clearOutputs(ctx: Context, keep: File? = null) {
        outDir(ctx).listFiles()?.forEach { if (it != keep) it.delete() }
    }

    /** 权重（8 KB）随包走，和模型不同 —— 它是我们自己训的，每次发版都可能变。 */
    private fun weightsPath(ctx: Context): String {
        val f = File(ctx.filesDir, "rerank.bin")
        if (!f.exists()) {
            ctx.assets.open("rerank.bin").use { i -> f.outputStream().use { o -> i.copyTo(o) } }
        }
        return f.absolutePath
    }

    private fun retriever(ctx: Context, uri: Uri) =
        MediaMetadataRetriever().apply { setDataSource(ctx, uri) }

    /**
     * 每段抽一帧当缩略图。**同步且慢**（一段约几十毫秒），调用方放后台。
     *
     * 三个不显然的地方：
     *  * **整批共用一个 retriever。** 每段新开一个要重新解析容器和建索引，
     *    十几段能差出一个数量级。
     *  * 取的是**段中点**而不是起点。起点常常正好是挥拍前的静止画面，
     *    十几张缩略图长得一模一样，等于没有。
     *  * OPTION_CLOSEST_SYNC 取最近的关键帧，不做精确定位 ——
     *    缩略图差几帧没人看得出来，精确定位却要解码整个 GOP。
     */
    fun thumbnails(ctx: Context, uri: Uri, clips: List<Clip>, px: Int = 160):
        List<android.graphics.Bitmap?> = retriever(ctx, uri).use { r ->
        clips.map { c ->
            val atUs = ((c.start + c.duration / 2) * 1_000_000).toLong()
            runCatching {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
                    // 直接要小图，别把 1080p 的帧解出来再缩 —— 十几段会很占内存
                    r.getScaledFrameAtTime(
                        atUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC, px, px)
                } else {
                    r.getFrameAtTime(atUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                }
            }.getOrNull()
        }
    }

    fun durationOf(ctx: Context, uri: Uri): Double = retriever(ctx, uri).use { r ->
        (r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L) / 1000.0
    }

    private fun sizeOf(ctx: Context, uri: Uri): Cutter.Canvas = retriever(ctx, uri).use { r ->
        val w = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 1280
        val h = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 720
        val rot = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull() ?: 0
        if (rot == 90 || rot == 270) Cutter.Canvas(h, w) else Cutter.Canvas(w, h)
    }

    private inline fun <T> MediaMetadataRetriever.use(block: (MediaMetadataRetriever) -> T): T =
        try { block(this) } finally { release() }

    /**
     * 存进相册。**用 MediaStore 而不是往公共目录写文件** ——
     * Android 10 起作用域存储生效，直接写路径在新系统上要么没权限、
     * 要么写进去了相册也扫不到。
     */
    fun saveToGallery(ctx: Context, src: File, displayName: String): Uri {
        val cv = ContentValues().apply {
            put(MediaStore.Video.Media.DISPLAY_NAME, displayName)
            put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Video.Media.RELATIVE_PATH,
                    Environment.DIRECTORY_MOVIES + "/Pipo")
                put(MediaStore.Video.Media.IS_PENDING, 1)
            }
        }
        val resolver = ctx.contentResolver
        val uri = resolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, cv)
            ?: throw RuntimeException("相册拒绝了写入")
        // insert 一旦成功，相册里就已经有了一条记录 —— **后面任何一步失败都必须
        // 把它删掉**，否则留下的是一个 0 字节的 .pending 条目。实测见过一次：
        // 成片被系统清了，copyTo 抛 ENOENT，相册里就多出
        // 「.pending-1786502965-Pipo_auto_….mp4」，用户看得见、删不掉，
        // 要等系统几天后自己回收。
        try {
            resolver.openOutputStream(uri)!!.use { o ->
                src.inputStream().use { i -> i.copyTo(o) }
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                cv.clear()
                cv.put(MediaStore.Video.Media.IS_PENDING, 0)
                resolver.update(uri, cv, null, null)
            }
        } catch (e: Throwable) {
            runCatching { resolver.delete(uri, null, null) }
            throw e
        }
        return uri
    }
}
