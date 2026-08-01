plugins {
    kotlin("jvm") version "2.1.0"
}

// 本机只有 JDK 21/25，没有 17。用 21 编译，但**字节码目标定在 17** ——
// Android 的 D8 不吃 21 的 class 文件版本，而这个模块将来要被 app 模块引用。
// 不用 jvmToolchain(17)：那会要求机器上真有一套 JDK 17。
java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}
kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

dependencies {
    // 桌面版 ORT：core 是纯 JVM 模块，这样 Rerank 的数值可以在 JVM 上
    // 直接对着 Python 比，不用模拟器。Android 侧用 onnxruntime-android，
    // 是同一个 ORT 内核的不同打包。
    compileOnly("com.microsoft.onnxruntime:onnxruntime:1.20.0")
    testImplementation("com.microsoft.onnxruntime:onnxruntime:1.20.0")
    testImplementation(kotlin("test"))
}

repositories { mavenCentral() }

tasks.test {
    useJUnitPlatform()
    testLogging {
        events("passed", "failed", "skipped")
        showStandardStreams = true      // 让 println 的对比数字露出来
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
    }
}
