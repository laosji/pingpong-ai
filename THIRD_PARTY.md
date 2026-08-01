# 第三方组件

本仓库的代码是 MIT（见 LICENSE）。运行和分发时会用到下面这些，
许可各不相同，尤其是 **ffmpeg 这一条会影响你怎么分发打包好的应用**。

## 模型权重

| | 来源 | 许可 |
|---|---|---|
| **PANNs CNN14**（`cnn14.onnx`） | [qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn) | MIT |

我们只是把它导出成 ONNX，网络结构和权重都是原作者的。
按需下载的那份托管在 R2 上，是同一份权重的 ONNX 形式。

`models/rerank.npz` 是本项目自己训的（2048 维逻辑回归，8 KB），
训练数据是 `labels/` 里的人工标注，随本仓库一起 MIT。

## ffmpeg —— 分发前请读这一段

Pipo 通过**子进程**调用 `ffmpeg` / `ffprobe`，不做链接。

仓库里**不包含** ffmpeg 二进制（`bin/` 已在 .gitignore）。你需要自己准备一份。

**注意构建配置。** 常见的 macOS 静态构建（例如 evermeet / tessus）带
`--enable-gpl --enable-version3`，也就是 **GPLv3**。如果你把这样的二进制
打进 .app 再分发：

* 以子进程方式调用通常被视为独立程序，不会让 Pipo 自身被传染为 GPL；
* 但你**分发的那个安装包里含有 GPLv3 程序**，因此需要随包提供 ffmpeg 的
  许可全文，并按 GPL 的要求提供对应源码或书面获取承诺。

想避开这件事，可以改用不带 `--enable-gpl` 的 LGPL 构建。代价是没有
libx264 —— macOS 上可以换成 `h264_videotoolbox`（系统自带的硬件编码器），
但这需要改 `ppai/render.py` 里的编码器参数，本项目**没有验证过那条路**。

**这不是法律意见。** 要商业分发请自己确认。

## Python 运行时依赖

numpy（BSD-3）、onnxruntime（MIT）、opencv-python-headless（Apache-2.0）、
FastAPI / Starlette / uvicorn（BSD-3 / MIT）、pywebview（BSD-3）、PyYAML（MIT）。

仅训练时用：PyTorch（BSD-3）、scikit-learn（BSD-3）、panns_inference（MIT）。
运行时用不到，打包时已排除。

## Android

Media3 / ExoPlayer（Apache-2.0）、ONNX Runtime Android（MIT）、Kotlin 标准库（Apache-2.0）。

## 测试素材

`android/` 下的测试音视频是 **`tools/make_fixture.py` 合成的**，
不含任何真人影像或录音。合成参数照着本项目实测的声学结构设定
（挥拍间隔、落台、弹跳衰减），所以仍能触发同样的代码路径 ——
实测覆盖比原来用的真实片段还好一些。
