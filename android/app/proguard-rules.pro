# onnxruntime 的 Java 层通过 JNI 反射拿类和字段，按名字找 —— 混淆会让它
# 在运行时抛 NoSuchMethodError，而且只在 release 包上复现。
-keep class ai.onnxruntime.** { *; }
-keepclassmembers class ai.onnxruntime.** { *; }

# media3 的 Transformer / ExoPlayer 有按名字实例化的组件
-keep class androidx.media3.** { *; }
-dontwarn androidx.media3.**

# 我们自己的算法层没有反射，可以正常混淆。
