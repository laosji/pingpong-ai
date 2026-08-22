# onnxruntime 的 Java 层通过 JNI 反射拿类和字段，按名字找 —— 混淆会让它
# 在运行时抛 NoSuchMethodError，而且只在 release 包上复现。
-keep class ai.onnxruntime.** { *; }
-keepclassmembers class ai.onnxruntime.** { *; }

# media3 的 Transformer / ExoPlayer 有按名字实例化的组件
-keep class androidx.media3.** { *; }
-dontwarn androidx.media3.**

# 我们自己的算法层没有反射，可以正常混淆。

# 但**自定义异常要留名字**。诊断报告里打的是 err.javaClass.name，
# 混淆之后传回来的是「异常 s1.l0」—— 而这几个类的名字本身就是诊断信息
# （NotPingpong 和 OutputGone 是完全不同的两件事，用户该做的事也不同）。
# 只留名字不留成员，混淆强度基本不受影响。
-keepnames class cc.pipo.app.Pipeline$* extends java.lang.Exception
-keepnames class cc.pipo.app.ModelStore$* extends java.lang.Exception
