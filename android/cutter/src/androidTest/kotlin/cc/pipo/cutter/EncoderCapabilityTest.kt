package cc.pipo.cutter

import android.media.MediaCodecInfo
import android.media.MediaCodecList
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * 问这台设备的 H.264 编码器：我们要输出的画布，你到底接不接受。
 *
 * 为什么需要这个
 * --------------
 * 一台华为机器卡在「拼接成片」，而我们查不出原因。首要嫌疑是
 * **画布尺寸不被厂商编码器接受** —— 而这类失败的表现常常是
 * **挂起而不是报错**，所以既不会抛异常、也不会进日志。
 *
 * 更要命的是我们**从没测到过**：手上的素材是 480x848、1024x512、544x960，
 * 全都恰好是 16 的倍数；而手机最常见的 1080p 竖屏，宽度 1080 % 16 = 8。
 *
 * 这个测试不跑真正的转码（那要几分钟、还要素材），只查能力表 ——
 * 秒级、无依赖，而且**在哪台机器上跑就给出哪台机器的答案**。
 * 华为那边只要跑一次，海思的真实数字就有了。
 *
 * 各家的名字完全不同：高通 OMX.qcom.*、海思 OMX.hisi.*、
 * 联发科 OMX.MTK.*、软件回退 c2.android.* / OMX.google.*。
 */
@RunWith(AndroidJUnit4::class)
class EncoderCapabilityTest {

    private data class Enc(
        val name: String, val hw: Boolean,
        val wAlign: Int, val hAlign: Int, val maxW: Int, val maxH: Int,
    )

    private fun encoders(): List<Enc> =
        MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos
            .filter { it.isEncoder && it.supportedTypes.any { t -> t.equals("video/avc", true) } }
            .mapNotNull { c ->
                val v = runCatching {
                    c.getCapabilitiesForType("video/avc").videoCapabilities
                }.getOrNull() ?: return@mapNotNull null
                Enc(c.name,
                    android.os.Build.VERSION.SDK_INT < 29 || c.isHardwareAccelerated,
                    v.widthAlignment, v.heightAlignment,
                    v.supportedWidths.upper, v.supportedHeights.upper)
            }

    @Test
    fun 打印这台设备的编码器能力() {
        val list = encoders()
        println("  设备 ${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}"
            + " / API ${android.os.Build.VERSION.SDK_INT}")
        list.forEach {
            println("  ${it.name}  ${if (it.hw) "硬件" else "软件"}"
                + "  对齐 ${it.wAlign}x${it.hAlign}  上限 ${it.maxW}x${it.maxH}")
        }
        assertTrue("这台设备一个 H.264 编码器都没有", list.isNotEmpty())
    }

    /**
     * **这是真正要验的那条。** pickCanvas 的输出必须被至少一个编码器接受，
     * 否则转码要么失败要么挂起。
     */
    @Test
    fun pickCanvas的输出所有编码器都能接受() {
        val list = encoders()
        val sources = listOf(
            1080 to 1920,   // 手机竖屏原生分辨率，宽度不是 16 的倍数
            1920 to 1080,
            2160 to 3840,   // 4K 竖屏
            3840 to 2160,
            720 to 1280,
            544 to 960,     // 我们的原片
            480 to 848,
        )
        val bad = ArrayList<String>()
        for ((sw, sh) in sources) {
            val c = Cutter.pickCanvas(listOf(Cutter.Canvas(sw, sh)))
            for (e in list) {
                val reasons = ArrayList<String>()
                if (c.width % e.wAlign != 0) reasons.add("宽 ${c.width} 不是 ${e.wAlign} 的倍数")
                if (c.height % e.hAlign != 0) reasons.add("高 ${c.height} 不是 ${e.hAlign} 的倍数")
                if (c.width > e.maxW || c.height > e.maxH) {
                    reasons.add("超过上限 ${e.maxW}x${e.maxH}")
                }
                if (reasons.isNotEmpty()) {
                    bad.add("源 ${sw}x$sh -> 画布 ${c.width}x${c.height}"
                        + " 在 ${e.name} 上：${reasons.joinToString("，")}")
                }
            }
            println("  ${sw}x$sh -> ${c.width}x${c.height}")
        }
        assertTrue("画布不被编码器接受：\n" + bad.joinToString("\n"), bad.isEmpty())
    }

    /**
     * 反向确认：**如果不做对齐，1080p 会不会真的出问题。**
     *
     * 这个测试不会失败 —— 它只是把「这台设备原本会不会踩到这个坑」
     * 打印出来。索尼的高通编码器对齐要求只有 2x2，所以在那台机器上
     * 不做对齐也没事；海思如果要求 16，答案就在这里。
     */
    @Test
    fun 报告不对齐时这台设备会不会拒绝() {
        for (e in encoders()) {
            val w = 1080
            val ok = w % e.wAlign == 0
            println("  ${e.name}：1080 % ${e.wAlign} = ${w % e.wAlign}"
                + if (ok) " —— 这台不受影响" else " —— **这台会踩到**")
        }
    }
}
