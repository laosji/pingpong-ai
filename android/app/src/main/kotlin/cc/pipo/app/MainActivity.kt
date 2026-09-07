package cc.pipo.app

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.media3.common.util.UnstableApi
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import java.io.File

/**
 * 四步：选视频 → 选主题 → 剪辑 → 保存。
 *
 * 做成线性四步而不是一个工作台，是因为手机上一屏放不下多个决定，
 * 而这条链路本来就是顺序的：没选素材谈不上选主题，没剪完谈不上保存。
 * 桌面版是工作台（可以来回改），手机版是流程。
 */
@UnstableApi
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ModelStore.cleanup(this)      // 清掉上次中断的半截下载
        // 上次剪到一半被系统杀掉的现场。**必须在这里传** —— 那种失败
        // 不抛异常，当时没人有机会上报，只有下次启动能捡回来。
        Diagnostics.sendLeftover(this)
        // 之前没传上去的（当时没网、或者传到一半进程没了）。一台真机报了
        // 「Muxer error」而这边什么都没收到，就是因为上报原本发一次就算。
        Diagnostics.flushOutbox(this)
        setContent { MaterialTheme(colorScheme = PipoDark) { App() } }
    }

    /**
     * 退到后台。**处理期间发生这件事很可能就是「卡住」的成因**，
     * 所以记进诊断时间线，见 [Diagnostics.onBackground]。
     */
    override fun onStop() {
        super.onStop()
        Diagnostics.onBackground()
    }
}

/**
 * Cut 是进度，Edit 是调片段，两者都属于进度条上的「剪辑」这一格 ——
 * 见 [StepBar] 的映射。分成两个状态而不是一个，是因为它们的界面
 * 毫无共同点：一个只有进度条，一个是播放器加一串可删的片段。
 */
private enum class Step { Pick, Theme, Cut, Edit, Done }

/**
 * 把异常翻成用户能看懂、而且**知道下一步该做什么**的一句话。
 *
 * **只有一处** —— 之前首次剪辑和删段重剪各有一套，同一个 OOM
 * 在前者有中文提示、在后者甩英文原文。这种分叉迟早会再长出来，
 * 除非只留一个入口。
 *
 * 原则：能说清「为什么」和「怎么办」的才翻；翻不了的宁可保留原文，
 * 也不要编一句听起来很友好但没信息的话 —— 原文至少还能截图发过来。
 */
private fun humanize(e: Throwable): String = base(e) + leftHint()

/**
 * 这一轮切走过就补一句。**放在最后而不是替换掉原因** ——
 * 切走不一定是这次失败的原因，但它是用户**唯一能自己避开**的那个因素，
 * 所以值得说；而把它说成原因就成了甩锅。
 */
private fun leftHint(): String =
    if (Diagnostics.leftDuringWork)
        "\n\n（这次处理中途切到了别的应用。剪辑期间系统可能会回收后台应用，" +
            "下次留在页面上试试。）"
    else ""

private fun base(e: Throwable): String = when (e) {
    // 系统抛的原文是「Failed to allocate a 139739088 byte
    // allocation with 25100288 free bytes and 55MB until OOM」——
    // 一句英文加一串字节数，用户既看不懂也不知道该做什么。
    // 实测就是这么甩到一位用户脸上的。
    is OutOfMemoryError ->
        "这段录像太长，这台手机的内存放不下。分成几段再剪就行 —— 十分钟以内比较稳。"
    else -> e.message ?: e.toString()
}

// CenterAlignedTopAppBar 目前仍是实验 API。显式 OptIn 而不是关掉整个警告 ——
// 关掉的话以后真有 API 变更也不会被提醒。
@OptIn(ExperimentalMaterial3Api::class)
@UnstableApi
@Composable
private fun App() {
    val ctx = androidx.compose.ui.platform.LocalContext.current
    val scope = rememberCoroutineScope()

    var step by remember { mutableStateOf(Step.Pick) }
    var uri by remember { mutableStateOf<Uri?>(null) }
    var durationS by remember { mutableStateOf(0.0) }
    var theme by remember { mutableStateOf(Pipeline.THEMES[0]) }
    var stage by remember { mutableStateOf<Pipeline.Stage?>(null) }
    // 「刚做完」的那几步攒成一条清单，剪辑页显示。只进不退 ——
    // 用户想看的是「已经查明了什么」，不是最新一条覆盖前一条。
    var facts by remember { mutableStateOf<List<String>>(emptyList()) }

    fun onStage(s: Pipeline.Stage) {
        stage = s
        // 打点。卡住时「卡在哪一步、前面几步花了多久」是第一手线索，
        // 而用户不会用 adb —— 不自己记就永远拿不到。
        //
        // 用 Stage.mark 而不是 javaClass.simpleName：后者在 release 包里
        // 会被 R8 改成 w0/v0/B0，见 Stage 的说明。
        Diagnostics.mark(s.mark)
        val line = when (s) {
            is Pipeline.Stage.Decoded -> "读完 %.0f 秒音轨".format(s.seconds)
            is Pipeline.Stage.Detected -> "听到 ${s.n} 次可能的击球"
            is Pipeline.Stage.Reranked ->
                "认定 ${s.kept} 次是真的，排除 ${s.total - s.kept} 次"
            is Pipeline.Stage.Grouped -> "凑出 ${s.rallies} 个回合，选中 ${s.clips} 段"
            else -> null
        }
        if (line != null) facts = facts + line
    }
    var dl by remember { mutableStateOf<ModelStore.Progress?>(null) }
    var result by remember { mutableStateOf<Pipeline.Result?>(null) }
    var saved by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }

    // 逐段删除：allClips 是分析出来的全集，removed 是用户划掉的下标。
    // **划掉而不是真删** —— 用户看完预览常会想把某段加回来，
    // 真删了就只能从头再分析一遍（那是整条链路里最慢的一步）。
    var allClips by remember { mutableStateOf<List<Pipeline.Clip>>(emptyList()) }
    var removed by remember { mutableStateOf<Set<Int>>(emptySet()) }
    var applied by remember { mutableStateOf<Set<Int>>(emptySet()) }   // 当前成片对应的删除集
    // 只保留画面的哪一半（把对手裁出画外）。和 removed 一样是「待应用」+
    // 「已应用」两份 —— 界面要据此判断当前成片是不是还对得上用户的选择。
    // 每段的头尾微调量（秒）。key 是 allClips 的下标，值是 (头, 尾)：
    // 头为负 = 往前多留，尾为正 = 往后多留。
    //
    // **为什么需要它**：切点来自击球声，而一个回合的开头（抛球、引拍）和
    // 结尾（球落地、捡球）本来就没有声音。固定留白（头 0.4 / 尾 0.7 秒）
    // 必然有时不够、有时多余，而每个回合都不一样 —— 这不是调常数能解决的。
    // 与其我去猜一个更好的常数，不如把这 0.1 秒交给看得见画面的人。
    var nudge by remember { mutableStateOf<Map<Int, Pair<Double, Double>>>(emptyMap()) }
    var nudgeApplied by remember { mutableStateOf<Map<Int, Pair<Double, Double>>>(emptyMap()) }
    var recutting by remember { mutableStateOf(false) }
    /**
     * 当前成片是不是最新的。
     *
     * **判据是「有没有未应用的改动」，不是「正在不正在切」。**
     * recutting 只在真正开切之后才为 true，而删片段后有 500 毫秒防抖窗口 ——
     * 那段时间里 recutting 还是 false，保存按钮可点，存进相册的却是
     * **删除前的旧成片**；等重切完成，clearOutputs 还会把刚才正在存的
     * 那份删掉。
     */
    val stale = removed != applied || recutting
    // 缩略图后填。**不挡着进调整页** —— 抽十几帧要一两秒，
    // 为了它把整页卡住不值得，先出卡片、图后到。
    var thumbs by remember { mutableStateOf<List<android.graphics.Bitmap?>>(emptyList()) }

    var job by remember { mutableStateOf<kotlinx.coroutines.Job?>(null) }
    var confirmAbort by remember { mutableStateOf(false) }
    // 这次的错误值不值得给「重试」按钮。只有网络中断算 ——
    // 「不是乒乓球」「空间不够」重试一百次也是同一个结果。
    var retryable by remember { mutableStateOf(false) }
    // 留着异常对象本身，不只是消息 —— 诊断信息里要类型和堆栈
    var lastError by remember { mutableStateOf<Throwable?>(null) }
    // 保存中。**必须挡住重复点击** —— 点两下会往相册存两份。
    var saving by remember { mutableStateOf(false) }

    // **处理期间屏幕不许灭。**
    //
    // 整条分析加剪辑跑在 Activity 的协程里 —— 没有前台服务，也没有唤醒锁。
    // 屏幕一灭，应用就退到后台，系统可以随时冻结或者杀掉这个进程；
    // EMUI 在这件事上比 AOSP 激进得多。用户放下手机等两分钟回来，
    // 看到的就是一个永远不动的进度条 —— 和那台华为机器报上来的现象一样。
    //
    // 常亮是最小的够用解：屏幕不灭 -> 应用一直在前台 -> 谁也不会杀它。
    // 真正完整的解是搬进前台服务（那样用户还能切去干别的），
    // 但那要加权限、通知渠道、API 33 的通知授权，改动大得多；
    // 而「放下手机等着」本来就是这个场景里绝大多数的情况。
    val view = androidx.compose.ui.platform.LocalView.current
    val busy = step == Step.Cut || recutting || saving
    DisposableEffect(busy) {
        view.keepScreenOn = busy
        Diagnostics.busy = busy
        onDispose { view.keepScreenOn = false; Diagnostics.busy = false }
    }

    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { picked ->
        if (picked != null) {
            // **读不出来就当场说，别放进流程。**
            // 原来是 getOrDefault(0.0)：读不到时长静默当成 0 秒，照样往下走，
            // 然后在分析阶段炸出一句和「选视频」毫无关系的话。
            // 相册返回的 uri 不一定能打开 —— 云端占位（还没下载回本机）、
            // 各家魔改相册返回的 content uri 我们没有权限、文件已被删，
            // 都会走到这里。这几种情况用户能自己解决，前提是我们说清楚。
            val d = runCatching { Pipeline.durationOf(ctx, picked) }.getOrDefault(0.0)
            if (d <= 0.0) {
                error = "这段视频读不出来。如果它还存在云端（相册里带云朵标记），" +
                    "先在相册里下载到手机再选；从别的应用分享过来的也可能打不开。"
                Diagnostics.autoSend(ctx, IllegalStateException("picked uri unreadable"))
                return@rememberLauncherForActivityResult
            }
            uri = picked
            error = null
            durationS = d
            step = Step.Theme
        }
    }

    fun start() {
        val u = uri ?: return
        step = Step.Cut
        error = null
        removed = emptySet(); applied = emptySet(); saved = false
        retryable = false
        // arm 而不是 reset：从这一刻起每个打点都会落盘，进程被系统杀掉时
        // 现场还留在磁盘上，下次启动捡回来传。收尾在下面的 finally。
        Diagnostics.arm(ctx)
        // 把进程钉在前台档，这样用户切走时不会掉进「缓存应用」档被顺手杀掉。
        // 界面上那句「请留在这一页」还留着 —— 前台服务不是万能的
        // （厂商省电策略、6 小时配额、起不来时静默失败），
        // 两条一起用：能扛住就扛住，扛不住至少用户知道该怎么避开。
        WorkService.start(ctx)
        facts = emptyList()          // 不清的话第二次剪会接在第一次的清单后面
        // 记住这个任务，用户放弃时才能真的取消 —— 不取消的话它会在后台
        // 接着跑完两分钟的分析，白白吃电和内存。
        job = scope.launch {
            try {
                withContext(Dispatchers.IO) {
                    // 上一次的成片留着没用了 —— 用户要么已经存进相册，
                    // 要么明确又剪了一次。不清的话剪十次会攒十份。
                    Pipeline.clearOutputs(ctx)
                    if (!ModelStore.ready(ctx)) {
                        ModelStore.download(ctx) { dl = it }
                        dl = null
                    }
                    // 阻塞代码看不见协程取消，得把状态传进去。
                    // **分析这一段原来是漏的** —— 只有下面的 cut 有，
                    // 于是「放弃」之后分析还在后台跑到底。
                    val clips = Pipeline.analyze(ctx, u, theme.id, onStage = ::onStage,
                        shouldStop = { !coroutineContext.isActive })
                    allClips = clips
                    // 把协程的取消状态传进去 —— 阻塞等待自己看不到
                    result = Pipeline.cut(ctx, u, clips, ::onStage) { !coroutineContext.isActive }
                }
                step = Step.Edit
                // 页面已经出来了，缩略图慢慢补
                scope.launch(Dispatchers.IO) {
                    thumbs = Pipeline.thumbnails(ctx, u, allClips)
                }
            } catch (e: kotlinx.coroutines.CancellationException) {
                throw e          // 用户主动放弃，不是错误，别弹出来
            } catch (e: Throwable) {
                // 放弃不是错误。**按类型判，不按文案判** —— 原来这里比的是
                // `e.message == "已取消"`，后来给失败消息包了一层人话，
                // message 变成「没能把片段拼成成片（已取消）。…」，比较失配，
                // 于是用户点了「放弃」反而弹出一个红色错误卡片。
                if (e is Pipeline.Cancelled) throw kotlinx.coroutines.CancellationException()
                error = humanize(e)
                lastError = e
                Diagnostics.autoSend(ctx, e)
                retryable = e is ModelStore.Interrupted
                step = Step.Theme
            } finally {
                stage = null; dl = null
                // 收工。**成功、报错、用户放弃都算收尾** —— 只有「进程没了」
                // 才走不到这里，而那正是我们要靠磁盘上那份现场捡回来的情况。
                // 报错那条上面已经 autoSend 过了，这里删掉避免下次启动重复传。
                Diagnostics.disarm()
                // 通知必须撤掉，否则会一直挂在状态栏上。
                // 和 disarm 一样放在 finally：三种收尾都要走到。
                WorkService.stop(ctx)
            }
        }
    }

    // 删/恢复之后重切。**只重切，不重新分析** —— 分析结果（allClips）没变。
    //
    // 500 毫秒的防抖：用户常会连着划掉三四段，每划一次就切一次的话
    // 前面几次全是白干，而切片要真跑一遍 Transformer。
    // keep 也是重剪的触发条件之一：换保留哪一半要重新走一遍 Transformer，
    // 和删段是同一类改动，共用同一套防抖和「待应用 / 已应用」判断。
    LaunchedEffect(removed, nudge, step) {
        val u = uri
        if (step != Step.Edit || u == null ||
            (removed == applied && nudge == nudgeApplied)) return@LaunchedEffect
        kotlinx.coroutines.delay(500)
        recutting = true
        error = null
        // 重剪同样是几分钟的活，一样要钉住进程 —— 用户删完段之后
        // 更可能顺手切走去干别的。
        WorkService.start(ctx)
        try {
            val adj = nudge
            val kept = allClips.mapIndexedNotNull { i, c ->
                if (i in removed) null else {
                    val (h, t) = adj[i] ?: (0.0 to 0.0)
                    Pipeline.Clip(c.start + h, c.end + t)
                }
            }
            val r = withContext(Dispatchers.IO) {
                Pipeline.cut(ctx, u, kept, { stage = it }) { !coroutineContext.isActive }
            }
            result = r
            applied = removed
            nudgeApplied = adj
            saved = false        // 存过的是上一版，这版还没存
            // 旧成片可以收了，但要留住正在放的这份
            withContext(Dispatchers.IO) { Pipeline.clearOutputs(ctx, keep = r.file) }
        } catch (e: kotlinx.coroutines.CancellationException) {
            throw e              // 换素材/回首页会取消这条协程，不是错误
        } catch (e: Pipeline.Cancelled) {
            // **这条以前整个漏了。** 主路径有取消判断，重剪路径一条都没有 ——
            // 在成片页删掉一段、重剪途中放弃，照样会弹红色错误卡片。
            removed = applied; nudge = nudgeApplied
        } catch (e: Throwable) {
            // **和主路径用同一套人话。** 之前这里是 e.message ?: e.toString()，
            // 同一个 OOM 在首次剪辑时有中文提示，在删掉一段重剪时却甩英文原文。
            error = humanize(e)
            Diagnostics.autoSend(ctx, e)   // 重剪失败一样要能查，之前漏了
            removed = applied    // 切失败就退回上一个能用的状态，别让界面和成片对不上
        } finally {
            WorkService.stop(ctx)
            recutting = false; stage = null
        }
    }

    val snackbar = remember { SnackbarHostState() }

    fun home() {
        step = Step.Pick; uri = null; result = null; saved = false
        allClips = emptyList(); removed = emptySet(); applied = emptySet(); error = null
        thumbs = emptyList()
    }

    fun back() {
        when (step) {
            Step.Theme -> { step = Step.Pick; uri = null; error = null }
            // **从「存好了」这一页返回就是回首页，不是退回调整页。**
            // 东西已经在相册里了，退回去会让人以为还没存 ——
            // 而且退回去再改的话，相册里会留下一份旧的。
            Step.Done -> home()
            Step.Edit -> home()
            // 剪辑中要问一句。分析要两三分钟，误触返回就全白跑了，
            // 而且系统返回手势从左边缘划，正好是最容易误触的地方。
            Step.Cut -> confirmAbort = true
            else -> {}
        }
    }

    // 接管系统返回键/返回手势。不接管的话剪辑中按返回会直接退出整个应用，
    // 没有确认、没有提示，两分钟的分析就没了。
    androidx.activity.compose.BackHandler(enabled = step != Step.Pick) { back() }

    fun save() {
        val r = result ?: return
        saving = true
        scope.launch {
            try {
                withContext(Dispatchers.IO) {
                    Pipeline.saveToGallery(ctx, r.file,
                        "Pipo_%s_%d.mp4".format(theme.id, System.currentTimeMillis()))
                }
                saved = true
                // 存完进预览页 —— 用户刚做完一件事，应该先看到成果，
                // 而不是被直接踢回首页。「再剪一个」在那一页上。
                step = Step.Done
                snackbar.showSnackbar("已存到相册的 Movies/Pipo")
            } catch (e: Throwable) {
                // 别把 java.io 的原文甩给用户 —— 实测这里出现过
                // 「/data/user/0/…/pipo_1785897055946.mp4: open failed:
                // ENOENT (No such file or directory)」，除了吓人没有任何用。
                error = if (e is java.io.FileNotFoundException ||
                    e.message?.contains("ENOENT") == true) {
                    "成片找不到了，多半是存储不够被系统清掉了。回去重剪一次。"
                } else {
                    "存不进相册：" + (e.message ?: e.javaClass.simpleName)
                }
            } finally {
                saving = false
            }
        }
    }

    Scaffold(
        snackbarHost = { SnackbarHost(snackbar) },
        topBar = {
            Column {
                CenterAlignedTopAppBar(
                    title = { Text("Pipo", fontWeight = FontWeight.SemiBold) },
                    // 返回在左，主动作在右 —— Material 的导航图标槽位就在左上角，
                    // 系统返回手势也从左边缘起，两者方向一致。
                    navigationIcon = {
                        if (step == Step.Theme || step == Step.Edit || step == Step.Done) {
                            IconButton({ back() }) {
                                Icon(Icons.AutoMirrored.Filled.ArrowBack, "返回")
                            }
                        }
                    },
                    // 「保存」放顶栏，不放在内容里 —— 分段列表可以很长，
                    // 跟在列表后面的按钮会被推到屏幕外，用户得先滚到底才点得到。
                    actions = {
                        if (step == Step.Edit && result != null) {
                            // **真的保存，然后进预览页。** 原来这个按钮写着
                            // 「保存」却只是跳到确认页 —— 标签在骗人，
                            // 用户以为存好了，其实还得在下一页再点一次。
                            TextButton({ save() }, enabled = !stale && !saving) {
                                Text(if (saving) "保存中" else "保存",
                                    fontWeight = FontWeight.SemiBold)
                            }
                        }
                    },
                )
                StepBar(step)
            }
        },
    ) { pad ->
        Column(
            Modifier.padding(pad).fillMaxSize().padding(20.dp)
                // **这两页自己管滚动，外层不能再套一层。**
                // verticalScroll 给子项的是无限高度约束，weight(1f) 在里面
                // 算出来是 0 —— 实测：给选主题页加了 weight 之后，
                // 四张主题卡直接消失，只剩底部按钮。
                //   Edit  要把视频撑满剩余高度
                //   Theme 要把「开始剪辑」钉在底部（小屏上会被推出屏幕）
                .then(if (step == Step.Edit || step == Step.Theme) Modifier
                      else Modifier.verticalScroll(rememberScrollState())),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            error?.let {
                Card(colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.errorContainer)) {
                    Column(Modifier.padding(14.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(it, fontSize = 13.sp,
                            color = MaterialTheme.colorScheme.onErrorContainer)
                        // 下载中断是**最常见的失败**（301 MB，中途断一次很正常），
                        // 而且它是可重试的 —— 让用户重走一遍选视频选主题不合理。
                        // 其他错误（不是乒乓球、空间不够）重试没有意义，不给按钮。
                        Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                            if (retryable && uri != null) {
                                TextButton({ error = null; start() },
                                    contentPadding = PaddingValues(0.dp)) {
                                    Text("重试下载", fontWeight = FontWeight.SemiBold,
                                        color = MaterialTheme.colorScheme.onErrorContainer)
                                }
                            }
                            // **每个错误都给。** 出问题时我们唯一能拿到的东西
                            // 就是用户愿意发过来的这段文字 —— 一台华为机器卡在
                            // 「拼接成片」，而我们什么都拿不到，只能问「再试一次？」
                            TextButton({
                                val cm = ctx.getSystemService(android.content.ClipboardManager::class.java)
                                cm?.setPrimaryClip(android.content.ClipData.newPlainText(
                                    "Pipo 诊断", Diagnostics.report(ctx, lastError)))
                                scope.launch { snackbar.showSnackbar("已复制，发给开发者即可") }
                            }, contentPadding = PaddingValues(0.dp)) {
                                Text("复制诊断信息",
                                    color = MaterialTheme.colorScheme.onErrorContainer)
                            }
                        }
                    }
                }
            }
            if (confirmAbort) {
                AlertDialog(
                    onDismissRequest = { confirmAbort = false },
                    title = { Text("放弃这次剪辑？") },
                    text = { Text("分析已经跑了一半，退出就要从头再来。") },
                    confirmButton = {
                        TextButton({
                            confirmAbort = false
                            job?.cancel()          // 真的停掉，别让它在后台空转
                            home()
                        }) { Text("放弃", color = MaterialTheme.colorScheme.error) }
                    },
                    dismissButton = {
                        TextButton({ confirmAbort = false }) { Text("继续剪") }
                    },
                )
            }
            when (step) {
                Step.Pick -> PickStep { picker.launch(
                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly)) }

                Step.Theme -> ThemeStep(
                    durationS = durationS, selected = theme, onSelect = { theme = it },
                    onBack = { step = Step.Pick; uri = null },
                    onStart = ::start,
                    // 包里带了模型就不该提示「要下 301 MB」
                    needsDownload = ModelStore.needsNetwork(ctx),
                )

                Step.Cut -> CutStep(stage, dl, facts, onCopyDiag = {
                    val cm = ctx.getSystemService(android.content.ClipboardManager::class.java)
                    cm?.setPrimaryClip(android.content.ClipData.newPlainText(
                        "Pipo 诊断", Diagnostics.report(ctx, null)))
                    scope.launch { snackbar.showSnackbar("已复制，发给开发者即可") }
                })

                Step.Edit -> EditStep(
                    modifier = Modifier.weight(1f),
                    result = result,
                    allClips = allClips, removed = removed, recutting = recutting,
                    thumbs = thumbs,
                    onToggle = { i ->
                        removed = if (i in removed) removed - i else removed + i
                    },
                    nudge = nudge,
                    onNudge = { i, dh, dt ->
                        val (h, t) = nudge[i] ?: (0.0 to 0.0)
                        val c = allClips[i]
                        // **不能越过邻段，也不能跑出素材。** 越过邻段的话
                        // 同一段画面会在成片里出现两次，而用户只会觉得「重复了」，
                        // 完全联想不到是自己多点了两下。
                        val prevEnd = allClips.getOrNull(i - 1)
                            ?.let { p -> p.end + (nudge[i - 1]?.second ?: 0.0) } ?: 0.0
                        val nextStart = allClips.getOrNull(i + 1)
                            ?.let { n -> n.start + (nudge[i + 1]?.first ?: 0.0) }
                            ?: Double.MAX_VALUE
                        val nh = (h + dh).let { v ->
                            (c.start + v).coerceIn(prevEnd, c.end + t - 0.3) - c.start
                        }
                        val nt = (t + dt).let { v ->
                            (c.end + v).coerceIn(c.start + nh + 0.3, nextStart) - c.end
                        }
                        nudge = if (kotlin.math.abs(nh) < 1e-6 && kotlin.math.abs(nt) < 1e-6)
                            nudge - i else nudge + (i to (nh to nt))
                    },
                )

                Step.Done -> DoneStep(
                    result = result,
                    // 「再剪一个」= 回到选素材那一步。和「回首页」现在是
                    // 同一个动作 —— 但保留两个按钮：措辞决定用户理解成
                    // 「继续做事」还是「结束」，这两种心情不一样。
                    onAgain = ::home, onHome = ::home)
            }
        }
    }
}

@Composable
private fun StepBar(step: Step) {
    val names = listOf("选视频", "选主题", "剪辑", "保存")
    // Cut（跑进度）和 Edit（调片段）都算「剪辑」这一格 ——
    // 不能用 Step.entries.indexOf，那样 Edit 会点亮「保存」。
    val idx = when (step) {
        Step.Pick -> 0
        Step.Theme -> 1
        Step.Cut, Step.Edit -> 2
        Step.Done -> 3
    }
    Row(Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 4.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        // **步骤条不用强调色。** 它是状态，不是动作。橙色留给「能点的东西」——
        // 之前步骤条、保存、主按钮、选中卡四处都是橙的，处处强调等于没有强调。
        // 走过的用白，没走到的用暗灰；当前这一步靠标题的白来区分。
        names.forEachIndexed { i, n ->
            val done = i <= idx
            Column(Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally) {
                Box(Modifier.fillMaxWidth().height(3.dp).background(
                    if (done) MaterialTheme.colorScheme.onSurface.copy(alpha = 0.85f)
                    else MaterialTheme.colorScheme.outlineVariant,
                    RoundedCornerShape(2.dp)))
                Spacer(Modifier.height(6.dp))
                Text(n, fontSize = 11.sp,
                    fontWeight = if (i == idx) FontWeight.Medium else FontWeight.Normal,
                    color = when {
                        i == idx -> MaterialTheme.colorScheme.onSurface
                        done -> MaterialTheme.colorScheme.onSurfaceVariant
                        else -> MaterialTheme.colorScheme.outline
                    })
            }
        }
    }
}

@Composable
private fun PickStep(onPick: () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("选一段训练录像", fontSize = 22.sp, fontWeight = FontWeight.SemiBold)
        Text("固定机位、能听见击球声就行。检测靠声音，竖屏和低分辨率都不影响剪得准不准。",
            fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.height(8.dp))
        Button(onPick, Modifier.fillMaxWidth().height(52.dp),
            colors = pipoButtonColors()) { Text("从相册选择") }
        Text("录像不会离开这台手机。", fontSize = 12.sp,
            color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun ThemeStep(
    durationS: Double, selected: Pipeline.Theme, onSelect: (Pipeline.Theme) -> Unit,
    onBack: () -> Unit, onStart: () -> Unit, needsDownload: Boolean,
) {
    // **主按钮钉在底部，不跟内容滚。**
    // 720x1280 的屏上四张主题卡加说明就占满了一屏，「开始剪辑」被推到
    // 屏幕外 —— 用户选完主题看不到下一步该点什么，得先猜到要往下滑。
    // 主动作永远不该需要滚动才够得着。
    Column(Modifier.fillMaxSize()) {
        Column(
            Modifier.weight(1f).verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text("要剪成什么", fontSize = 22.sp, fontWeight = FontWeight.SemiBold)
            Text("素材 %d:%02d".format(durationS.toInt() / 60, durationS.toInt() % 60),
                fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Pipeline.THEMES.forEach { t ->
                Card(
                    Modifier.fillMaxWidth().selectable(t == selected) { onSelect(t) },
                    colors = CardDefaults.cardColors(
                        containerColor = if (t == selected)
                            MaterialTheme.colorScheme.primaryContainer
                        else MaterialTheme.colorScheme.surfaceVariant),
                ) {
                    Column(Modifier.padding(14.dp)) {
                        // 未选中的标题也用 onSurface —— 之前跟着容器色走成了
                        // onSurfaceVariant，四个选项里三个都是灰的，看着像被禁用
                        Text(t.name, fontWeight = FontWeight.Medium,
                            color = if (t == selected)
                                MaterialTheme.colorScheme.onPrimaryContainer
                            else MaterialTheme.colorScheme.onSurface)
                        Text(t.desc, fontSize = 12.sp,
                            color = if (t == selected)
                                MaterialTheme.colorScheme.onPrimaryContainer
                            else MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
            }
            if (needsDownload) {
                // 提前说，别等用户点了才弹一个 300MB 的下载出来
                Text("第一次剪辑需要下载约 %d MB 的声学模型，之后离线可用。"
                        .format(ModelStore.DOWNLOAD_BYTES / 1_000_000),
                    fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Spacer(Modifier.height(4.dp))
        }
        // 「换一段」删掉了 —— 顶栏左上角的返回做的就是这件事，
        // 同一个动作在一屏里出现两次，用户会以为它们不一样。
        Button(onStart, Modifier.fillMaxWidth().height(54.dp),
            colors = pipoButtonColors()) {
            Sparkle(MaterialTheme.colorScheme.surface)
            Spacer(Modifier.width(8.dp))
            Text("开始剪辑", fontWeight = FontWeight.SemiBold)
        }
    }
}

/**
 * 主按钮上的火花。
 *
 * 用四角星而不是引一套图标库：`AutoAwesome` 在 material-icons-extended 里，
 * 为一个 18dp 的形状拖进整包图标不值得，而这个形状是纯几何。
 *
 * 两颗大小不同的星 —— 一颗孤零零的容易读成「收藏」，成对才读得出
 * 「自动生成」那层意思。
 */
@Composable
private fun Sparkle(color: Color) {
    Canvas(Modifier.size(18.dp)) {
        fun star(cx: Float, cy: Float, r: Float) {
            // 四角星：四个方向各拉一个尖，腰部收到 0.34 —— 收得越紧越尖锐
            val w = r * 0.34f
            val p = androidx.compose.ui.graphics.Path().apply {
                moveTo(cx, cy - r)
                cubicTo(cx + w * 0.4f, cy - w * 0.4f, cx + w * 0.4f, cy - w * 0.4f, cx + r, cy)
                cubicTo(cx + w * 0.4f, cy + w * 0.4f, cx + w * 0.4f, cy + w * 0.4f, cx, cy + r)
                cubicTo(cx - w * 0.4f, cy + w * 0.4f, cx - w * 0.4f, cy + w * 0.4f, cx - r, cy)
                cubicTo(cx - w * 0.4f, cy - w * 0.4f, cx - w * 0.4f, cy - w * 0.4f, cx, cy - r)
                close()
            }
            drawPath(p, color)
        }
        star(size.width * 0.42f, size.height * 0.44f, size.minDimension * 0.42f)
        star(size.width * 0.82f, size.height * 0.80f, size.minDimension * 0.20f)
    }
}

@Composable
private fun CutStep(stage: Pipeline.Stage?, dl: ModelStore.Progress?, facts: List<String>,
                    onCopyDiag: () -> Unit) {
    // 已耗时。**卡住时用户最需要知道的是「这正常吗」** ——
    // 进度条不动加上没有任何时间信息，看着就像死机，而实际上
    // Transformer 在转码时本来就可能长时间不更新进度。
    var elapsed by remember { mutableStateOf(0) }
    val ctx0 = androidx.compose.ui.platform.LocalContext.current
    LaunchedEffect(Unit) {
        while (true) {
            kotlinx.coroutines.delay(1000)
            elapsed++
            // **卡住时也要传。** 那台华为就是卡在「拼接成片」，
            // 从来没抛异常，所以走不到错误分支 —— 只等异常的话
            // 最需要现场的那一类失败恰好一份都收不到。
            // 三分钟发一次，最多两次，够定位又不会变成心跳。
            if (elapsed == 180 || elapsed == 480) Diagnostics.autoSend(ctx0, null)
        }
    }
    Column(verticalArrangement = Arrangement.spacedBy(18.dp)) {
        Text("正在剪辑", fontSize = 22.sp, fontWeight = FontWeight.SemiBold)

        val (label, pct) = when {
            dl is ModelStore.Progress.Downloading ->
                "下载声学模型 %d / %d MB".format(dl.done / 1_000_000, dl.total / 1_000_000) to
                    (100f * dl.done / dl.total.coerceAtLeast(1)).toInt()
            dl is ModelStore.Progress.Unpacking ->
                "准备声学模型 %d / %d MB".format(dl.done / 1_000_000, dl.total / 1_000_000) to
                    (100f * dl.done / dl.total.coerceAtLeast(1)).toInt()
            dl is ModelStore.Progress.Finishing -> "校验模型" to -1
            stage is Pipeline.Stage.Decoding -> "读取音轨" to -1
            stage is Pipeline.Stage.Detecting -> "找出所有像击球的声音" to -1
            stage is Pipeline.Stage.Reranking -> "分辨哪些是真的击球" to stage.percent
            stage is Pipeline.Stage.Cutting -> "拼接成片" to stage.percent
            else -> "准备中" to -1
        }

        BouncingProgress(pct)

        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(label, fontSize = 15.sp, fontWeight = FontWeight.Medium)
            Spacer(Modifier.weight(1f))
            if (pct >= 0) {
                Text("$pct%", fontSize = 15.sp, fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.primary)
            }
        }

        // 攒出来的事实清单。等两三分钟的时候，「它在干什么」比「还要多久」
        // 更能让人安心 —— 而且这些数字本来就有，不显示是浪费。
        if (facts.isNotEmpty()) {
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                facts.forEach { f ->
                    Row(verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Box(Modifier.size(5.dp).clip(CircleShape)
                            .background(MaterialTheme.colorScheme.primary))
                        Text(f, fontSize = 13.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
            }
        }

        // **别切走。** 实测（模拟器，AOSP）：按 HOME 之后处理确实还在跑，
        // 后台 60 秒 CPU tick 涨了 3663，比前台还快 —— 因为不用渲染界面了。
        // 但同一时刻 oom_score_adj 从 0 跳到 **900**，也就是「缓存应用」档：
        // 内存一紧张就是第一批被杀的，而被杀时不抛异常、没有任何提示，
        // 用户看到的就是「回来一看，什么都没了」。
        // 模拟器有 16GB 所以扛过去了，手机不一定 —— EMUI 尤其激进。
        //
        // 所以这句不是客套，是**真的会丢东西**。放在耗时提示前面，
        // 用 primary 色，和下面那句灰色的隐私说明区分开。
        Row(verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("请留在这一页", fontSize = 12.sp,
                color = MaterialTheme.colorScheme.primary)
            Text("切到别的应用可能会被系统中断，得从头再来",
                fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }

        Text("已用 %d:%02d · 全程在这台手机上完成，录像不会上传。"
                .format(elapsed / 60, elapsed % 60),
            fontSize = 12.sp, color = MaterialTheme.colorScheme.outline)

        // 三分钟还没完就说句话。45 秒素材实测两分钟左右，超过这个
        // 多半是这台机器的编解码器有问题 —— 而那正是诊断信息能回答的。
        if (elapsed > 180) {
            Card(colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
                Column(Modifier.padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text("比预期久了一些。素材越长越慢，但如果进度很久不动，"
                        + "可能是这台手机的编解码器和我们的组合有问题。",
                        fontSize = 12.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                    TextButton(onCopyDiag, contentPadding = PaddingValues(0.dp)) {
                        Text("复制诊断信息发给开发者", fontSize = 12.sp)
                    }
                }
            }
        }
    }
}

/**
 * 一颗球沿着进度条弹过去。
 *
 * 用球而不是转圈：这个产品从头到尾就是在听球的声音，
 * 让等待的那两分钟也在讲同一件事。
 *
 * 拿得到百分比时球停在对应位置（并且一直在原地弹，说明没卡死）；
 * 拿不到时球来回横穿 —— **不编一个假的百分比**，
 * 但也不能让画面完全静止，那和卡死分不出来。
 */
@Composable
private fun BouncingProgress(percent: Int) {
    val t = rememberInfiniteTransition(label = "ball")
    // 一个弹跳周期。用 abs(sin) 得到「落地-弹起-落地」的形状，
    // 顶点圆滑、落点尖锐，和真实弹跳一致
    val phase by t.animateFloat(
        0f, 1f, infiniteRepeatable(tween(560, easing = LinearEasing), RepeatMode.Restart),
        label = "phase")
    // 不定态时的横向往返
    val sweep by t.animateFloat(
        0f, 1f, infiniteRepeatable(tween(1600, easing = FastOutSlowInEasing),
            RepeatMode.Reverse),
        label = "sweep")

    val lift = kotlin.math.abs(kotlin.math.sin(phase * Math.PI)).toFloat()
    val frac = if (percent >= 0) (percent / 100f).coerceIn(0f, 1f) else sweep

    val track = MaterialTheme.colorScheme.outlineVariant
    val fill = MaterialTheme.colorScheme.primary
    val ball = MaterialTheme.colorScheme.primary

    Canvas(Modifier.fillMaxWidth().height(46.dp)) {
        val r = 7.dp.toPx()
        val baseY = size.height - r
        val x0 = r
        val x1 = size.width - r
        // 轨道
        drawLine(track, Offset(x0, baseY), Offset(x1, baseY),
            strokeWidth = 3.dp.toPx(), cap = StrokeCap.Round)
        val x = x0 + (x1 - x0) * frac
        if (percent >= 0 && frac > 0f) {
            drawLine(fill, Offset(x0, baseY), Offset(x, baseY),
                strokeWidth = 3.dp.toPx(), cap = StrokeCap.Round)
        }
        // 落点的影子：球越高影子越淡越小，这样「弹」才读得出来
        drawOval(
            color = track.copy(alpha = 0.55f * (1f - lift)),
            topLeft = Offset(x - r * (1f - lift * 0.4f), baseY + 2.dp.toPx()),
            size = Size(2 * r * (1f - lift * 0.4f), 3.dp.toPx()),
        )
        drawCircle(ball, r, Offset(x, baseY - lift * (size.height - 3 * r)))
    }
}

private fun mmss(s: Double): String =
    "%d:%02d".format(s.toInt() / 60, s.toInt() % 60)

/** 调整页：看预览、去掉不想要的段。顶栏的「完成」进确认页。 */
@androidx.annotation.OptIn(UnstableApi::class)
@Composable
private fun EditStep(
    modifier: Modifier,
    result: Pipeline.Result?,
    allClips: List<Pipeline.Clip>, removed: Set<Int>, recutting: Boolean,
    thumbs: List<android.graphics.Bitmap?>,
    onToggle: (Int) -> Unit,
    nudge: Map<Int, Pair<Double, Double>>,
    /** (段下标, 头部增量秒, 尾部增量秒)。负数=往前，正数=往后。 */
    onNudge: (Int, Double, Double) -> Unit,
) {
    val keptCount = allClips.size - removed.size
    val player = rememberPlayer(result?.file)

    // 成片的宽高比，播放器按它定高度。
    //
    // **读文件元数据，不轮询 player.videoSize** —— 后者实测拿不到值：
    // 画面明明已经在放了，videoSize 却一直是 0，于是比例算成 0、
    // 播放器退回「吃满剩余高度」，上下两条大黑边。成片文件就在手上，
    // 一次读准比盯着播放器状态可靠。
    var aspect by remember(result?.file) { mutableStateOf(0f) }
    LaunchedEffect(result?.file) {
        val f = result?.file ?: return@LaunchedEffect
        aspect = withContext(Dispatchers.IO) { Pipeline.aspectOf(f) }
    }

    // 播放位置。**必须轮询** —— ExoPlayer 没有「位置变了」的回调
    // （它是连续量，事件化没有意义）。150 毫秒够跟上眼睛，
    // 又不至于每帧都触发重组。
    var posMs by remember { mutableStateOf(0L) }
    LaunchedEffect(player) {
        while (true) {
            posMs = player.currentPosition
            kotlinx.coroutines.delay(150)
        }
    }

    // 留下来的段，连同它们在 allClips 里的原下标 —— 高亮要标在原下标上。
    val kept = remember(allClips, removed) {
        allClips.withIndex().filter { it.index !in removed }
    }
    // 成片实际时长和「各段时长之和」会差一点（切点要对齐帧），
    // 直接按累加时长算边界会越到后面偏得越多。按比例缩一下就对齐了。
    val reqSum = kept.sumOf { it.value.duration }
    val durMs = player.duration
    val scale = if (durMs > 0 && reqSum > 0) durMs / (reqSum * 1000.0) else 1.0

    // 每段在成片里的起点（毫秒）
    val starts = remember(kept, scale) {
        var acc = 0.0
        kept.map { (_, c) -> val s = acc * scale * 1000; acc += c.duration; s.toLong() }
    }
    val activeIdx = kept.indices.lastOrNull { posMs + 1 >= starts[it] } ?: -1
    val activeOriginal = if (activeIdx >= 0) kept[activeIdx].index else -1

    // 画面优先：标题去掉了，视频吃掉所有剩余高度，片段条贴着底。
    // 这一页用户是在「看」，不是在「读」—— 每让出一行文字，
    // 画面就大一点。段数和体积压成一行小字跟在下面。
    // 选框模式，以及拖动中的草稿。草稿单独存是为了「取消」能原样退回，
    // 也为了拖动过程不触发重剪。
    Column(modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        // **盒子跟着画面比例走，不要给一个固定的大高度。**
        //
        // 原来是 `PlayerBox(player, Modifier.weight(1f))`：盒子吃掉所有剩余
        // 高度，而 RESIZE_MODE_FIT 只会把画面等比塞进去 —— 横屏成片放进一个
        // 很高的盒子，上下就是两条大黑边，画面看着「特别小」。
        //
        // 这里显式算尺寸，不用 aspectRatio 修饰符：那个要么宽度优先、
        // 要么高度优先，另一个方向就可能冲破约束（竖屏比例下算出的高度
        // 超过可用空间，把下面的内容全顶出屏幕 —— 这个坑踩过，
        // 见 PlayerBox 的注释）。BoxWithConstraints 里两边都夹住，
        // 横屏竖屏都不会溢出。
        if (result != null) {
            BoxWithConstraints(Modifier.weight(1f).fillMaxWidth()) {
                val w = maxWidth
                val h = maxHeight
                // **宽度永远铺满，高度跟着比例走** —— 短视频平台都是这个样子：
                // 画面顶着两边，容器紧贴画面，没有多余的黑框。
                //
                // 只有一种情况要退让：竖屏成片按宽度算出来的高度会超过可用空间，
                // 那时改成高度吃满、宽度让步，否则会把下面的片段条顶出屏幕
                // （这个坑踩过，见 PlayerBox 的注释）。
                val th = if (aspect > 0f) minOf(w / aspect, h) else h
                val tw = if (aspect > 0f) minOf(th * aspect, w) else w
                // 贴顶而不是居中：横屏成片上方会空出一大块，居中会让画面
                // 悬在中间，短视频平台都是顶着上沿的。
                PlayerBox(player, Modifier.size(tw, th).align(Alignment.TopCenter))
            }
        }

        Row(verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            result?.let {
                Text("%d 段 · %s · %.1f MB".format(
                    it.clips.size, mmss(it.seconds), it.file.length() / 1e6),
                    fontSize = 13.sp, fontWeight = FontWeight.Medium)
            }
            Spacer(Modifier.weight(1f))
            if (recutting) {
                CircularProgressIndicator(Modifier.size(13.dp), strokeWidth = 2.dp)
                Text("重剪中", fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            } else {
                Text("点卡片跳到那一段", fontSize = 11.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }

        // 选中那一段的头尾微调。
        //
        // **只对正在播的那一段显示**：十几张卡片每张都挂四个按钮，
        // 会把片段条挤成一团按钮，而用户一次只调一段。
        //
        // 步长 0.1 秒：再小听不出差别，再大就跳过了想要的那一帧。
        activeOriginal.takeIf { it >= 0 && it in allClips.indices && it !in removed }
            ?.let { i ->
                val (h, t) = nudge[i] ?: (0.0 to 0.0)
                Row(verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text("第 ${i + 1} 段", fontSize = 12.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                    Spacer(Modifier.width(4.dp))
                    Text("开头", fontSize = 12.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                    NudgeBtn("−0.1", !recutting) { onNudge(i, -0.1, 0.0) }
                    NudgeBtn("+0.1", !recutting) { onNudge(i, 0.1, 0.0) }
                    Spacer(Modifier.width(8.dp))
                    Text("结尾", fontSize = 12.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                    NudgeBtn("−0.1", !recutting) { onNudge(i, 0.0, -0.1) }
                    NudgeBtn("+0.1", !recutting) { onNudge(i, 0.0, 0.1) }
                    if (h != 0.0 || t != 0.0) {
                        Spacer(Modifier.weight(1f))
                        Text("%+.1f / %+.1f".format(h, t), fontSize = 11.sp,
                            color = MaterialTheme.colorScheme.primary)
                    }
                }
            }

        if (allClips.isNotEmpty()) {
            // **横排而不是竖排。** 一段常常只有 1-2 秒，十几段竖着列下来
            // 能占三四屏，播放器和按钮全被挤出视野；横排一屏能看到三四段，
            // 而且「一段接一段」的排布本来就更像时间轴。
            val listState = rememberLazyListState()
            // 播到哪一段就把它滚进视野 —— 高亮在屏幕外等于没有高亮
            LaunchedEffect(activeOriginal) {
                val at = allClips.indices.indexOf(activeOriginal)
                if (at >= 0) runCatching { listState.animateScrollToItem(at) }
            }
            LazyRow(
                state = listState,
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                contentPadding = PaddingValues(vertical = 2.dp),
            ) {
                items(allClips.size) { i ->
                    val c = allClips[i]
                    val gone = i in removed
                    val playing = i == activeOriginal
                    // 只剩最后一段时不许再删 —— 空成片没有意义，
                    // 而且 Cutter 收到空列表会直接抛异常。
                    val canRemove = !gone && keptCount > 1
                    // 整张卡就是那一帧画面，文字压在上面。
                    // 「播放中」不靠底色区分了（底色被画面盖住），改成描边。
                    Box(
                        Modifier.width(132.dp).height(112.dp)
                            .clip(RoundedCornerShape(12.dp))
                            .background(MaterialTheme.colorScheme.surfaceVariant)
                            .then(
                                if (playing) Modifier.border(
                                    2.dp, MaterialTheme.colorScheme.primary,
                                    RoundedCornerShape(12.dp))
                                else Modifier
                            )
                            .let { m ->
                                // 划掉的段不在成片里，点它没有可跳的位置
                                if (gone) m else m.selectable(playing) {
                                    val k = kept.indexOfFirst { it.index == i }
                                    if (k >= 0) { player.seekTo(starts[k]); player.play() }
                                }
                            }
                    ) {
                        thumbs.getOrNull(i)?.let { bmp ->
                            androidx.compose.foundation.Image(
                                bmp.asImageBitmap(), "第 %d 段的画面".format(i + 1),
                                // 删掉的段整体压暗，一眼能看出它不在成片里
                                Modifier.fillMaxSize().alpha(if (gone) 0.25f else 1f),
                                contentScale = ContentScale.Crop,
                            )
                        }
                        // 压一层从下往上的暗角。**没有它文字必然会翻车** ——
                        // 乒乓球台是浅蓝加白线，白字直接压上去有一半看不见。
                        Box(
                            Modifier.fillMaxSize().background(
                                androidx.compose.ui.graphics.Brush.verticalGradient(
                                    0.0f to Color.Black.copy(alpha = 0.45f),
                                    0.35f to Color.Transparent,
                                    1.0f to Color.Black.copy(alpha = 0.85f),
                                )
                            )
                        )
                        Column(
                            Modifier.align(Alignment.BottomStart)
                                .padding(start = 10.dp, end = 10.dp, bottom = 8.dp)
                        ) {
                            Text("%.1f 秒".format(c.duration), fontSize = 18.sp,
                                fontWeight = FontWeight.SemiBold, color = Color.White)
                            // 原片里的位置，方便用户对着源素材回想这一段
                            Text("%s → %s".format(mmss(c.start), mmss(c.end)),
                                fontSize = 10.sp, color = Color.White.copy(alpha = 0.75f))
                        }
                        // 角标给一个实底药丸。渐变暗角挡不住所有画面 ——
                        // 这段素材的天花板是白的，紫色的「播放中」压上去
                        // 几乎读不出来。底色不依赖画面内容，才是稳的。
                        Text(
                            when { gone -> "已删除"; playing -> "播放中"; else ->
                                "第 %d 段".format(i + 1) },
                            Modifier.align(Alignment.TopStart)
                                .padding(start = 8.dp, top = 8.dp)
                                .background(
                                    Color.Black.copy(alpha = 0.55f),
                                    RoundedCornerShape(5.dp))
                                .padding(horizontal = 6.dp, vertical = 2.dp),
                            fontSize = 10.sp, fontWeight = FontWeight.SemiBold,
                            color = when {
                                gone -> Color.White.copy(alpha = 0.7f)
                                playing -> MaterialTheme.colorScheme.primary
                                else -> Color.White.copy(alpha = 0.9f)
                            },
                        )
                        // 和角标同一个道理：17dp 的图标压在花画面上，
                        // 渐变暗角挡不住。给一个实底圆，可读性就不再取决于
                        // 这一帧恰好是什么内容。
                        IconButton(
                            { onToggle(i) },
                            Modifier.align(Alignment.TopEnd)
                                .size(38.dp)
                                // **不做成「选中才显示」。** 删片段是这一屏的主任务，
                                // 而且常要连删好几段；藏起来之后每删一段要两下
                                // （先选中再删），首次使用的人也看不出能删。
                                // 改成压弱：未选中时半透明、不画遮盖，选中时才全强度。
                                // 一下删除的成本保住了，一排卡片也不再是五个红点。
                                .alpha(if (playing) 1f else 0.45f)
                                .drawBehind {
                                    if (!playing) return@drawBehind
                                    // 遮盖是贴着圆角的扇形，从角上向内发散地淡出：
                                    // 圆形底衬要盖住整整一圈，而图标其实只需要它
                                    // 正下方那一小块有对比度。扇形面积不到圆的一半，
                                    // 边缘又是渐变，不会在缩略图上糊出一个硬块。
                                    val r = size.minDimension
                                    drawArc(
                                        brush = Brush.radialGradient(
                                            0f to Color.Black.copy(alpha = 0.72f),
                                            0.55f to Color.Black.copy(alpha = 0.5f),
                                            1f to Color.Transparent,
                                            center = Offset(size.width, 0f),
                                            radius = r,
                                        ),
                                        startAngle = 90f, sweepAngle = 90f, useCenter = true,
                                        topLeft = Offset(size.width - r, -r),
                                        size = Size(2 * r, 2 * r),
                                    )
                                },
                            enabled = (gone || canRemove) && !recutting,
                        ) {
                            if (gone) {
                                Icon(Icons.Filled.Refresh, "恢复这一段",
                                    Modifier.size(18.dp), tint = Color.White)
                            } else {
                                Icon(Icons.Filled.Delete, "删除这一段",
                                    Modifier.size(18.dp),
                                    // 只剩一段时按钮是禁用的，颜色也要跟着灰掉，
                                    // 否则看着可点却点不动
                                    tint = if (canRemove) MaterialTheme.colorScheme.error
                                           else Color.White.copy(alpha = 0.35f))
                            }
                        }
                    }
                }
            }
            if (keptCount <= 1) {
                Text("至少要留一段。", fontSize = 11.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

/**
 * 确认页：只有最终效果和两个去处。
 *
 * **故意不再列片段** —— 调整页刚列过一遍，这里再列一次就是同一个页面
 * 出现两次，用户会怀疑自己是不是没点动。这一页只回答一个问题：
 * 「就这样了吗？」
 */
@androidx.annotation.OptIn(UnstableApi::class)
@Composable
private fun DoneStep(result: Pipeline.Result?, onAgain: () -> Unit, onHome: () -> Unit) {
    val player = rememberPlayer(result?.file)
    // 和剪辑页同一套：宽度铺满、高度跟着成片比例。
    // **这一页之前漏了** —— 还用着 PlayerBox 的默认「固定 300dp 高」，
    // 竖屏成片塞进去就是中间一小块，四周全是黑。
    var aspect by remember(result?.file) { mutableStateOf(0f) }
    LaunchedEffect(result?.file) {
        val f = result?.file ?: return@LaunchedEffect
        aspect = withContext(Dispatchers.IO) { Pipeline.aspectOf(f) }
    }
    Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
        Text("存好了", fontSize = 22.sp, fontWeight = FontWeight.SemiBold)
        Text("在相册的 Movies/Pipo 里", fontSize = 13.sp,
            color = MaterialTheme.colorScheme.primary)
        if (result != null) {
            BoxWithConstraints(Modifier.fillMaxWidth()) {
                val w = maxWidth
                // 这一页没有 weight，高度不受限，所以只按宽度算，
                // 但给一个上限免得竖屏成片把按钮顶到屏幕外。
                val th = if (aspect > 0f) minOf(w / aspect, 520.dp) else 300.dp
                val tw = if (aspect > 0f) minOf(th * aspect, w) else w
                PlayerBox(player, Modifier.size(tw, th).align(Alignment.Center))
            }
        }
        result?.let {
            Text("%d 段 · %s · %.1f MB".format(
                it.clips.size, mmss(it.seconds), it.file.length() / 1e6),
                fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Spacer(Modifier.height(4.dp))
        // 「再剪一个」是主动作：刚剪完一段的人最可能想剪下一段，
        // 而不是关掉应用。
        Button(onAgain, Modifier.fillMaxWidth().height(52.dp),
            colors = pipoButtonColors()) {
            Sparkle(MaterialTheme.colorScheme.surface)
            Spacer(Modifier.width(8.dp))
            Text("再剪一个", fontWeight = FontWeight.SemiBold)
        }
        OutlinedButton(onHome, Modifier.fillMaxWidth().height(48.dp)) { Text("回首页") }
    }
}

/**
 * 建一个播放器，绑到 [file]，并保证离开时释放。
 *
 * **释放这件事不能忘** —— 桌面版在这里栽过：剪完之后原视频还在后台播，
 * 用户听得见声音却找不到源头。DisposableEffect 兜住所有离开路径，
 * 不管是点「返回」还是被系统回收。
 *
 * 每次重剪都是一个新文件，所以换片挂在 [file] 上而不是 Unit ——
 * 挂 Unit 的话删完一段预览还停在旧成片上，用户会以为删除没生效。
 */
@androidx.annotation.OptIn(UnstableApi::class)
@Composable
private fun rememberPlayer(file: File?): androidx.media3.exoplayer.ExoPlayer {
    val ctx = androidx.compose.ui.platform.LocalContext.current
    val player = remember { androidx.media3.exoplayer.ExoPlayer.Builder(ctx).build() }
    DisposableEffect(Unit) { onDispose { player.release() } }
    LaunchedEffect(file) {
        if (file == null) return@LaunchedEffect
        player.setMediaItem(androidx.media3.common.MediaItem.fromUri(Uri.fromFile(file)))
        player.prepare()
    }
    return player
}

/** 微调用的小圆钮。做得小而密 —— 它们会被连点，手指移动距离越短越好。 */
@Composable
private fun NudgeBtn(label: String, enabled: Boolean, onClick: () -> Unit) {
    Surface(
        onClick = onClick, enabled = enabled,
        shape = RoundedCornerShape(50),
        color = MaterialTheme.colorScheme.surfaceVariant,
    ) {
        Text(label, fontSize = 12.sp,
            modifier = Modifier.padding(horizontal = 9.dp, vertical = 4.dp),
            color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@androidx.annotation.OptIn(UnstableApi::class)
@Composable
private fun PlayerBox(
    player: androidx.media3.exoplayer.ExoPlayer,
    modifier: Modifier = Modifier.height(300.dp),
) {
    // **固定高度，比例交给 PlayerView 自己处理。**
    //
    // 别写成 `fillMaxWidth().heightIn(max = 320.dp).aspectRatio(a)`：
    // aspectRatio 默认先按宽度定高（matchHeightConstraintsFirst = false），
    // 竖屏成片比例 0.566，算出来的高度是宽度的 1.77 倍 —— 直接冲破前面的
    // heightIn，视频溢出布局，下面的文字全被压在画面上。实测踩过。
    //
    // 给一个确定的高度、让 RESIZE_MODE_FIT 在里面等比缩，是唯一
    // 不依赖修饰符顺序的写法。竖屏两侧留黑边，和所有视频应用一致。
    Box(
        modifier.fillMaxWidth()
            .clip(RoundedCornerShape(12.dp)).background(Color.Black)
    ) {
        AndroidView(
            factory = { c ->
                androidx.media3.ui.PlayerView(c).apply {
                    this.player = player
                    useController = true
                    resizeMode = androidx.media3.ui.AspectRatioFrameLayout.RESIZE_MODE_FIT
                    setShutterBackgroundColor(android.graphics.Color.BLACK)
                    // 只留「后退 5 / 播放 / 前进 15」。上一个/下一个在这里
                    // 永远是灰的 —— 播放列表里只有一支成片，点了不会有任何反应，
                    // 摆着只会让人以为坏了。
                    setShowPreviousButton(false)
                    setShowNextButton(false)
                    setShowSubtitleButton(false)
                    setShowShuffleButton(false)
                }
            },
            modifier = Modifier.fillMaxSize(),
        )
    }
}
