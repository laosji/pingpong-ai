// 只声明 core：它是纯 JVM 模块，不依赖 Android，所以单测不用模拟器就能跑。
// app 模块（Android）等 core 验证通过再加 —— 地基没验完就往上砌墙，
// 出错时分不清是移植错了还是 Android 那层的问题。
pluginManagement {
    repositories { google(); gradlePluginPortal(); mavenCentral() }
    plugins {
        id("com.android.library") version "8.7.3"
        kotlin("android") version "2.1.0"
    }
}
dependencyResolutionManagement {
    repositories { google(); mavenCentral() }
}
rootProject.name = "pipo"
include(":core")
include(":cutter")
