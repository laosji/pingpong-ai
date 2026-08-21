plugins {
    id("com.android.library")
    kotlin("android")
}

android {
    namespace = "cc.pipo.cutter"
    compileSdk = 35
    defaultConfig {
        // MediaMuxer/MediaCodec 的关键能力从 26 起才齐。
        //
        // **这条在真机上验过了，不要再降。** 2025-08-21 在 Sony E5803
        // （Android 7.1.1 / API 25、骁龙 810）上把 minSdk 探到 24：
        // 界面、模型下载、解码、检测、重排全部正常（45 秒素材分析只用 50 秒），
        // 但 Transformer 起来配好 codec 之后**永久挂起** —— 进度停在 16%，
        // 应用日志此后 8 分钟一个字都没有，只能等 30 分钟超时。
        // 降 minSdk 换来的不是「多支持一批老机器」，是「多一批卡死的用户」。
        minSdk = 26
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        // ONNX Runtime 每个 ABI 一份原生库，四个 ABI 全打进去测试 APK 就是 80MB。
        // 正式发版应该用 ABI splits 或 AAB，让每台设备只下自己那份。
        ndk { abiFilters += listOf("arm64-v8a") }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    // 测试素材放 androidTest/assets，源码目录用 kotlin/ 而不是默认的 java/
    sourceSets {
        getByName("main") { kotlin.srcDir("src/main/kotlin") }
        getByName("androidTest") {
            kotlin.srcDir("src/androidTest/kotlin")
            assets.srcDir("src/androidTest/assets")
        }
    }
}

dependencies {
    api(project(":core"))          // 仪器测试要直接用 core 里的 Audio
    // Media3 Transformer：Google 官方的转码/剪辑库。
    // 不手搓 MediaCodec 的理由：切片要精确到帧（关键帧吸附会让片段提前
    // 1-3 秒开始，实测手机录像的关键帧间隔就是 1s 和 3s），
    // 而精确切割意味着重编码，重编码意味着要处理一大堆设备兼容问题。
    // Transformer 已经把这些处理掉了，还带硬件加速。
    implementation("androidx.media3:media3-transformer:1.5.1")
    implementation("androidx.media3:media3-effect:1.5.1")
    api("androidx.media3:media3-common:1.5.1")   // @UnstableApi 注解要透给 app 模块
    // ONNX Runtime：跑 PANNs CNN14 出嵌入。**只用 fp32 模型** ——
    // int8/fp16 都测过，前者让前五片段只剩 55% 重合且更慢，
    // 后者改掉 28% 的选段而优劣未验证（见 README）。
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")

    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("junit:junit:4.13.2")
}
