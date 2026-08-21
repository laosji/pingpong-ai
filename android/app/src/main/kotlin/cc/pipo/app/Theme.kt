package cc.pipo.app

import androidx.compose.material3.ColorScheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.ui.graphics.Color

/**
 * 配色。
 *
 * 换掉 Compose 的 `darkColorScheme()` 默认值 —— 那个紫（#D0BCFF）是
 * Material 3 的出厂基线色，含义就是「作者没有选颜色」，所有跳过主题配置的
 * Compose 应用长得一模一样，而紫色和乒乓球没有任何关系。
 *
 * 强调色取**球的橙**：它是这个产品的主角，在深底上够暖够亮。
 * 为什么不用蓝或红 —— 画面里已经有大片的蓝台面和红地胶，
 * 强调色撞上去就糊在一起了；橙色比地胶的暗红亮得多，不会混。
 *
 * 深底保留。视频工具的界面就该后退，让画面说话；而且底色带一点暖偏移
 * （#12100D 而不是纯灰黑），和橙色是同一支色温，不会显得强调色是贴上去的。
 *
 * error 特意选了明确偏红的 #FF5C5C 而不是 M3 默认的粉：删除图标只有 17dp，
 * 和橙色强调色摆在同一张卡片上，靠明度分不开，得靠色相。
 */
private val BallOrange = Color(0xFFFFB067)      // 强调：球
private val OnBallOrange = Color(0xFF4A2400)    // 橙底上的字
private val OrangeDeep = Color(0xFF6B3A08)      // 选中态容器
private val OnOrangeDeep = Color(0xFFFFDCC0)

private val Ink = Color(0xFF12100D)             // 页面底
private val InkRaised = Color(0xFF1C1814)       // 卡片底
private val InkCard = Color(0xFF2A241E)         // surfaceVariant
private val Chalk = Color(0xFFEFE3D6)           // 正文
private val ChalkDim = Color(0xFFCBBCAB)        // 次要文字
private val Edge = Color(0xFF574C41)

val PipoDark: ColorScheme = darkColorScheme(
    primary = BallOrange,
    onPrimary = OnBallOrange,
    primaryContainer = OrangeDeep,
    onPrimaryContainer = OnOrangeDeep,

    // 次强调用同一支暖色的低饱和版本，不引入第二个色相 ——
    // 一个界面里两个抢眼的颜色会互相削弱
    secondary = Color(0xFFD8C3AC),
    onSecondary = Color(0xFF3A2E20),
    secondaryContainer = Color(0xFF524535),
    onSecondaryContainer = Color(0xFFF5E0C8),

    background = Ink,
    onBackground = Chalk,
    surface = Ink,
    onSurface = Chalk,
    surfaceVariant = InkCard,
    onSurfaceVariant = ChalkDim,
    surfaceContainer = InkRaised,
    outline = Edge,
    outlineVariant = Color(0xFF3A322A),

    error = Color(0xFFFF5C5C),
    onError = Color(0xFF48090A),
    errorContainer = Color(0xFF7A1D1C),
    onErrorContainer = Color(0xFFFFDAD6),
)

/**
 * 主按钮：白底深字。
 *
 * **动作用白，状态用橙。** 主按钮是一屏里唯一「往前走」的东西，
 * 白色在近黑底上的对比度谁也压不过；橙色留给选中态、播放中、顶栏保存
 * 这些「现在是什么情况」的信号。两者分工不同，各占一个颜色，
 * 就不会出现「四处都在强调」的问题。
 *
 * 不直接把配色里的 primary 改成白：那样选中卡的边框、播放中角标
 * 也会跟着变白，和按钮抢同一个视觉重量。
 */
@androidx.compose.runtime.Composable
fun pipoButtonColors() = androidx.compose.material3.ButtonDefaults.buttonColors(
    containerColor = Color(0xFFF5F1EC),          // 带一点暖的白，和橙色同色温
    contentColor = Ink,
    disabledContainerColor = Color(0xFF2E2822),
    disabledContentColor = Color(0xFF6E6459),
)
