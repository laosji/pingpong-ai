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
        minSdk = 26
        targetSdk = 35
        versionCode = 2
        versionName = "0.1.1"
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
    // 把声学模型打进安装包。**只用于直接分发，不能上商店** ——
    // 模型压完 288 MB，加上本体是 307 MB，而 APK 上限 100 MB、
    // AAB 基础模块 150 MB，都过不去。商店版仍然走按需下载。
    //
    //   ./gradlew assembleRelease -PpipoBundleModel
    //
    // 生成的 assets 文件不进版本库（见 .gitignore）。
    if (project.hasProperty("pipoBundleModel")) {
        val src = rootProject.file("../models/cnn14.onnx")
        require(src.exists()) { "要打包模型得先有 models/cnn14.onnx" }
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
