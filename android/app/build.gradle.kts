import java.io.FileInputStream
import java.util.Properties

plugins {
    id("com.android.application")
    kotlin("android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "cc.pipo.app"
    compileSdk = 35
    defaultConfig {
        applicationId = "cc.pipo.app"
        // **只在真机排查时临时下探**，用 -PpipoMinSdk=25，不改默认值。
        // 手上唯一一台带高通硬件编解码器的机器（Sony E5803）是 API 25，
        // 而模拟器只有软件编解码器 —— 有些问题只在厂商实现上出现，
        // 不下探就一次都测不到。cutter 模块有同样的开关。
        // 正式发版不带这个参数，仍然是 26（依据见 cutter/build.gradle.kts）。
        minSdk = (project.findProperty("pipoMinSdk") as String?)?.toInt() ?: 26
        targetSdk = 35
        versionCode = 14
        versionName = "0.2.3"
        // ONNX Runtime 每个 ABI 一份原生库。
        // **不要在这里写死 abiFilters** —— 之前只留 arm64-v8a，
        // armeabi-v7a 和 x86_64 的机器直接装不上。正式发版走 AAB，
        // Play 会按设备下发对应那一份，包体不受影响。
        // 只在本地调试时用 -PpipoAbi=arm64-v8a 收窄，加快构建。
        (project.findProperty("pipoAbi") as String?)?.let { a ->
            ndk { abiFilters += a.split(",") }
        }
    }
    // 签名信息从 keystore.properties 读，**那个文件不进版本库**
    // （见 .gitignore）。缺文件时 release 仍然能构建，只是不签名 ——
    // 这样 CI 和别人 clone 下来跑 assembleRelease 不会直接失败。
    val ksProps = Properties()
    rootProject.file("keystore.properties").let { f ->
        if (f.exists()) FileInputStream(f).use { ksProps.load(it) }
    }
    signingConfigs {
        if (ksProps.getProperty("storeFile") != null) {
            create("release") {
                storeFile = rootProject.file(ksProps.getProperty("storeFile"))
                storePassword = ksProps.getProperty("storePassword")
                keyAlias = ksProps.getProperty("keyAlias")
                keyPassword = ksProps.getProperty("keyPassword")
            }
        }
    }
    buildTypes {
        getByName("release") {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"),
                          "proguard-rules.pro")
            signingConfig = signingConfigs.findByName("release")
        }
        getByName("debug") {
            // 装在同一台机器上不覆盖正式版
            applicationIdSuffix = ".debug"
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    // buildConfig：诊断信息里要报版本号和构建类型，
    // AGP 8 起默认关闭，得显式打开
    buildFeatures { compose = true; buildConfig = true }

    // 出错时自动把诊断信息传回来。**只给自己人测试的包用。**
    //
    //   ./gradlew assembleRelease -PpipoAutoDiag
    //
    // 做成开关而不是常开：正式版必须先问过用户才能传任何东西
    // （PIPL、各家商店的要求，而且界面上写着「全程在这台手机上完成」）。
    // 默认 false 意味着**忘了加这个参数的包不会偷偷上传** ——
    // 反过来（默认开、发版记得关）迟早会有一次忘记。
    defaultConfig {
        buildConfigField("boolean", "AUTO_DIAG",
            if (project.hasProperty("pipoAutoDiag")) "true" else "false")
    }
    sourceSets {
        getByName("main") {
            kotlin.srcDir("src/main/kotlin")
            assets.srcDir("src/main/assets")
        }
    }
    // 把声学模型打进安装包。**默认就打包** —— 我们现在只做本地版。
    //
    // 原来是反过来的：要显式加 -PpipoBundleModel 才打包，忘了加就打出一个
    // 「必须联网下载」的包，而下载地址至今是 404 —— 也就是一个装上去
    // 根本用不了的包，而且要到用户点「开始剪辑」才暴露。
    // **默认值应该是那个「忘了也不会错」的选项。**
    //
    // 想做商店版（走按需下载）时显式关掉：
    //   ./gradlew assembleRelease -PpipoNoBundleModel
    // 那种包才需要 R2 上有模型；带模型的包 288 MB，超过 APK 100 MB /
    // AAB 基础模块 150 MB 的上限，上不了商店，只用于直接分发。
    //
    // 生成的 assets 文件不进版本库（见 .gitignore）。
    if (!project.hasProperty("pipoNoBundleModel")) {
        val src = rootProject.file("../models/cnn14.onnx")
        require(src.exists()) {
            "本地版必须带模型，但找不到 models/cnn14.onnx。\n" +
                "要打不带模型的商店版，显式加 -PpipoNoBundleModel。"
        }
        // **放原始文件，不要放 .gz。** 试过放 gz：AGP 会把它解开、
        // 去掉后缀，APK 里的条目变成 assets/cnn14.onnx，
        // 而代码去找 cnn14.onnx.gz 找不到，静默回落到下载 —— 打包等于白做。
        // 直接放原始文件反而更简单：zip 自己的 deflate 压到同样的 288 MB，
        // 读的时候 AssetManager 透明解压，代码里一行 gzip 都不用写。
        val stagedDir = layout.buildDirectory.dir("bundled-assets")
        val stage = tasks.register<Copy>("stageBundledModel") {
            from(src); into(stagedDir)
        }
        // **必须挂给所有读这个目录的任务**，不能只挂 mergeAssets ——
        // lintVitalAnalyzeRelease 也会读它，漏了 Gradle 会直接报
        // 「uses this output without declaring an explicit dependency」。
        sourceSets.getByName("main").assets.srcDir(stage)
    }

    packaging { resources { excludes += "/META-INF/{AL2.0,LGPL2.1}" } }
}

dependencies {
    implementation(project(":cutter"))
    implementation(platform("androidx.compose:compose-bom:2025.01.00"))
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-core")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
    // 成片预览。版本必须和 cutter 里的 media3-transformer 一致 ——
    // media3 各模块共用 common，混版本会在运行时抛 NoSuchMethodError。
    implementation("androidx.media3:media3-exoplayer:1.5.1")
    implementation("androidx.media3:media3-ui:1.5.1")
    debugImplementation("androidx.compose.ui:ui-tooling")
}
