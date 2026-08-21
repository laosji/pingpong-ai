// 只声明 core：它是纯 JVM 模块，不依赖 Android，所以单测不用模拟器就能跑。
// app 模块（Android）等 core 验证通过再加 —— 地基没验完就往上砌墙，
// 出错时分不清是移植错了还是 Android 那层的问题。
// 仓库顺序：先镜像，后官方源。
//
// **不是为了「国内加速」这么简单** —— repo.maven.apache.org 对
// org.codehaus.groovy:groovy:3.0.21（AGP 的 lint-gradle 间接依赖）
// 在这条网络上直接返回 403，assembleRelease 会因此构建不出来，
// 而报错信息指向的是 :cutter:extractReleaseAnnotations，
// 完全看不出是仓库的问题。镜像放在前面，两个问题一起解决。
// 官方源保留在后面兜底：镜像缺件时还能回落。
pluginManagement {
    repositories {
        maven("https://maven.aliyun.com/repository/public")
        maven("https://maven.aliyun.com/repository/gradle-plugin")
        google(); gradlePluginPortal(); mavenCentral()
    }
    plugins {
        id("com.android.library") version "8.7.3"
        id("com.android.application") version "8.7.3"
        kotlin("android") version "2.1.0"
        id("org.jetbrains.kotlin.plugin.compose") version "2.1.0"
    }
}
dependencyResolutionManagement {
    repositories {
        maven("https://maven.aliyun.com/repository/public")
        maven("https://maven.aliyun.com/repository/google")
        google(); mavenCentral()
    }
}
rootProject.name = "pipo"
include(":core")
include(":cutter")
include(":app")
