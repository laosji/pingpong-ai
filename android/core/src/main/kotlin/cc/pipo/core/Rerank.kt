package cc.pipo.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.roundToInt

/**
 * 候选重排 —— ppai/rerank.py 的 Kotlin 版。
 *
 * 干的事：谱通量检测器给出的候选里混着一半以上的假货（球台碰撞、脚步、
 * 说话），用 PANNs CNN14 的嵌入 + 一个逻辑回归把它们排个序，按比例砍掉后半。
 * 实测把准确率从 0.38-0.49 提到 0.58-0.68。
 *
 * 两个必须照搬的约束
 * ------------------
 *  * **按比例保留，不能用固定概率阈值。** 概率标定不跨视频：实测 B→A 用
 *    固定阈值 0.5 时召回只剩 0.170，而按比例保留 50% 时是 0.755。
 *    排序能力跨视频，绝对概率值不跨。
 *  * **模型必须是 fp32。** 量化过的版本测过：int8 让前五片段只剩 55% 重合
 *    而且更慢，fp16 改掉 28% 的选段且优劣未验证。只有 fp32 与 torch 逐行一致。
 *
 * 为什么在 core 而不是 cutter
 * ---------------------------
 * 它一行 Android API 都不用（只有 ai.onnxruntime 和标准库）。放在纯 JVM 的
 * core 里，就能用桌面版的 onnxruntime 直接对着 Python 逐点比对，
 * 不必为了跑一个数值测试去装模拟器、推 323MB 模型、和存储空间搏斗。
 * Android 侧只剩「原生库能不能在 ARM 上加载」这一件事要验，那是另一回事。
 *
 * 内存
 * ----
 * 嵌入本身不大（1 秒窗 / 0.5 秒跳，一小时素材约 7200×2048×4 = 59MB），
 * **真正吃内存的是 PCM**：一小时 32kHz float 就是 460MB。手机上必须分段解码
 * 处理，见 [MAX_SAFE_SECONDS]。
 */
object Rerank {

    private const val PANNS_SR = 32000
    private const val WIN = 32000        // 1 秒
    private const val HOP = 16000        // 0.5 秒
    private const val BATCH = 64
    // 教师（CNN14）给 2048 维嵌入；蒸馏出来的学生直接给 1 维 logit。
    // **维度不写死，跟着权重文件走** —— 两个模型走的是同一条代码路径，
    // 学生配的 rerank.bin 是 coef=[1.0] / b=0，于是 probability() 算出来
    // 正好是 sigmoid(logit)，门禁的 0.10 和重排的 0.6 都不用重新标定。
    private const val TEACHER_DIM = 2048

    /**
     * 单次整段处理的时长上限。
     *
     * **这个数原来写死 20 分钟，而且是照着错的量算的。** 当时只算了
     * 32kHz PCM（每秒 128KB），漏掉了解码阶段 —— 那时候解码是先把整段按
     * **原始采样率**攒起来再拼成一个大数组，48kHz 单声道每秒 192KB，
     * 而且要同时拿着两份。12 分钟的真实素材实际峰值 279MB，
     * 从 20 分钟的守卫底下大摇大摆过去，然后在用户手机上 OOM。
     *
     * 解码改成流式之后（见 AudioDecode），峰值就是结果本身。现在按
     * **这台设备真实的堆上限**算，而不是拍一个所有机器通用的数：
     * 同样一段素材，512MB 堆的旗舰能剪，128MB 堆的老机器不能，
     * 写死一个数只能二选一 —— 要么冤枉前者，要么坑死后者。
     *
     * 峰值构成（一秒素材）：
     *   * 32kHz float PCM        128KB   嵌入这一步要整段
     *   * 嵌入 2 窗 × 2048 × 4    16KB
     * 合计约 144KB/秒。留一半余量给运行时自己、界面和缩略图。
     */
    fun maxSafeSeconds(maxHeapBytes: Long): Int {
        val perSecond = 144L * 1024
        val usable = (maxHeapBytes / 2).coerceAtLeast(32L * 1024 * 1024)
        // 上限仍然封在 30 分钟：再长的话即使内存够，等待时间也不合理了
        return (usable / perSecond).toInt().coerceIn(60, 30 * 60)
    }

    class Model(
        internal val env: OrtEnvironment,
        internal val session: OrtSession,
        internal val coef: FloatArray,
        internal val intercept: Float,
        internal val inputName: String,
    ) : Closeable {
        /** 嵌入宽度。教师 2048，学生 1 —— 由权重文件决定，不由代码假设。 */
        internal val dim: Int get() = coef.size

        override fun close() { runCatching { session.close() } }
    }

    /**
     * 载入模型。[onnxPath] 是 cnn14.onnx（323MB，按需下载到本机），
     * [weightsPath] 是 rerank.bin（8KB，跟着安装包走）。
     */
    fun load(onnxPath: String, weightsPath: String): Model {
        val env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))
        }
        // 传路径而不是字节数组：ORT 会 mmap 权重，避免把 323MB 读进堆里
        val session = env.createSession(onnxPath, opts)
        val (coef, b) = readWeights(File(weightsPath))
        return Model(env, session, coef, b, session.inputNames.iterator().next())
    }

    /**
     * 读 rerank.bin：魔数 "PIPO" + 版本(u32) + 维度(i32) + 截距(f32) + 系数(f32×n)。
     * 自定义格式而不是 .npz —— 那是 zip 套 npy，为 8KB 权重实现两层解析不值得。
     */
    internal fun readWeights(f: File): Pair<FloatArray, Float> {
        RandomAccessFile(f, "r").use { raf ->
            val head = ByteArray(16)
            raf.readFully(head)
            val hb = ByteBuffer.wrap(head).order(ByteOrder.LITTLE_ENDIAN)
            val magic = ByteArray(4).also { hb.get(it) }
            require(String(magic) == "PIPO") { "不是 rerank.bin：魔数不对" }
            val ver = hb.int
            require(ver == 1) { "rerank.bin 版本 $ver 不认识" }
            val n = hb.int
            val b = hb.float
            require(n == TEACHER_DIM || n == 1) {
                "系数维度 $n：只认 $TEACHER_DIM（CNN14 教师）或 1（蒸馏学生）"
            }
            val raw = ByteArray(4 * n)
            raf.readFully(raw)
            val fb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            return FloatArray(n) { fb.get() } to b
        }
    }

    /**
     * 滑窗提嵌入。返回 (窗中心时刻, [N][2048])。
     *
     * [onProgress] 收到 (已完成窗数, 总窗数)。**这一步占整条链路的绝大部分时间**
     * （45 秒素材上约 150 秒，而解码 + 检测 + 切片加起来才几十秒），
     * 不报进度的话界面上就是两分多钟的不定态转圈，用户没法判断是慢还是卡死。
     * 窗数在开始前就算得出来，所以这里给的是真百分比，不是编的。
     */
    fun embed(
        pcm32k: FloatArray, m: Model,
        onProgress: ((Int, Int) -> Unit)? = null,
        /**
         * 用户放弃了没有。**每一批查一次。**
         *
         * 这里是整条链路最长的一段（一份真机报告里 166 秒），而原来
         * 分析阶段一个取消检查都没有 —— 点了「放弃」界面退回去了，
         * 这个循环还在后台跑到底。实测放弃后 40 秒 CPU tick 仍在涨，
         * 而且比前台还快（不用渲染界面）。
         *
         * 查的粒度是「批」而不是「窗」：一批几十毫秒，够及时了，
         * 而每窗查一次是白白多几万次调用。
         */
        shouldStop: (() -> Boolean)? = null,
    ): Pair<DoubleArray, Array<FloatArray>> {
        var x = pcm32k
        if (x.size < WIN) x = x.copyOf(WIN)          // 和 Python 的 np.pad 一致：补零
        val n = 1 + (x.size - WIN) / HOP
        val times = DoubleArray(n) { (it.toLong() * HOP + WIN / 2.0) / PANNS_SR }
        val feats = Array(n) { FloatArray(m.dim) }

        var i = 0
        while (i < n) {
            if (shouldStop?.invoke() == true) {
                throw java.util.concurrent.CancellationException("已取消")
            }
            val b = minOf(BATCH, n - i)
            val buf = FloatBuffer.allocate(b * WIN)
            for (k in 0 until b) buf.put(x, (i + k) * HOP, WIN)
            buf.rewind()
            OnnxTensor.createTensor(m.env, buf, longArrayOf(b.toLong(), WIN.toLong())).use { t ->
                m.session.run(mapOf(m.inputName to t)).use { r ->
                    @Suppress("UNCHECKED_CAST")
                    val out = r[0].value as Array<FloatArray>
                    for (k in 0 until b) System.arraycopy(out[k], 0, feats[i + k], 0, m.dim)
                }
            }
            i += b
            onProgress?.invoke(i, n)
        }
        return times to feats
    }

    /** sigmoid(x·coef + b)。先裁再取指数，否则大负值那侧会溢出成 NaN。 */
    fun probability(f: FloatArray, m: Model): Double {
        var z = m.intercept.toDouble()
        for (j in 0 until m.dim) z += f[j].toDouble() * m.coef[j]
        return 1.0 / (1.0 + exp(-z.coerceIn(-700.0, 700.0)))
    }

    /**
     * 按比例保留候选。返回**按时间升序**的保留结果，和 Python 的 apply 一致。
     */
    fun apply(
        hits: DoubleArray, times: DoubleArray, feats: Array<FloatArray>,
        m: Model, keepRatio: Double = 0.6,
    ): DoubleArray {
        if (hits.isEmpty() || times.isEmpty()) return hits
        val p = DoubleArray(hits.size) { i ->
            // 每个候选取时间上最近的那个窗 —— 和 Python 的 argmin|t-c| 相同
            var best = 0
            var bd = Double.MAX_VALUE
            for (j in times.indices) {
                val d = abs(times[j] - hits[i])
                if (d < bd) { bd = d; best = j }
            }
            probability(feats[best], m)
        }
        val k = (hits.size * keepRatio).roundToInt().coerceAtLeast(1)
        val idx = p.indices.sortedWith(compareByDescending<Int> { p[it] }.thenBy { it }).take(k)
        return idx.map { hits[it] }.sorted().toDoubleArray()
    }

    /** 整段素材的平均概率 —— 用来判断「这是不是乒乓球录像」。 */
    fun meanProbability(feats: Array<FloatArray>, m: Model): Double {
        if (feats.isEmpty()) return 0.0
        var s = 0.0
        for (f in feats) s += probability(f, m)
        return s / feats.size
    }
}
