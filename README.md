# Pipo — 乒乓球训练录像自动剪辑

把一小时的训练录像，剪成几十秒的集锦。**全部在本机运行，录像不上传。**

固定机位拍一段，自动找出有效回合、剪掉捡球和等待。检测靠的是**击球声**，
不是画面——所以竖屏、低分辨率、逆光都不影响剪得准不准。

```bash
.venv/bin/python -m ppai.cli highlight 训练.mp4 --type power --top 5
```

或者用桌面版：拖进去，选主题，剪。

---

## 它能做什么，不能做什么

这一节写在最前面，因为**这个项目最容易出的问题是声称做到了没做到的事**。

**能做，且有实测支撑：**

| | 实测 |
|---|---|
| 剪掉捡球和等待，一个球都不漏 | 12 分钟压到 4:12，召回 0.927 |
| 按「单拍最响」挑扣杀 | 排序相关 0.89，最可靠的一项 |
| 按「来回最多」挑相持 | 留出视频 Spearman ρ 0.906 |
| 判断素材是对打还是多球训练 | 8 个视频（含 2 个留出）全判对 |
| 拒收不是乒乓球的录像 | 阴性 0.009-0.010 / 真实 0.379-0.672，差 40 倍 |

**做不到，也不打算假装：**

- **判断谁赢谁输**。声音里没有胜负信息——一方的绝杀就是另一方的失误。
- **球的落点和速度**。30fps 下球一帧走 0.56 米。
- **谁打的这一板**。音频无法归属到人。
- **只有一拍的回合**。单次击球和球台碰撞在声学上分不开（准确率 0.13）。
- **准确的拍数和回合数**。检测器会多数 2.7 倍——所以界面上**不显示**这些数字，
  只显示排序（排序是可靠的，绝对量不是）。

## 怎么跑

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

需要 `ffmpeg` / `ffprobe` 在 PATH 上（见 [THIRD_PARTY.md](THIRD_PARTY.md) 关于许可的说明）。

第一次运行需要声学模型（约 300 MB，只下一次）：

```bash
.venv/bin/python -c "from ppai import assets; assets.download(assets.ASSETS[0])"
```

常用命令：

```bash
ppai.cli probe   视频.mp4          # 素材质量与建议
ppai.cli analyze 视频.mp4          # 检测 + 出图 + JSON
ppai.cli highlight 视频.mp4 --type all
ppai.cli annotate 视频.mp4 --window 120 240   # 打开标注页面
ppai.cli rerank-train              # 用 labels/ 重训重排器
```

所有阈值在 `config.yaml`，可用 `-s audio.k_mad=6.0` 临时覆盖。

## 桌面版

```bash
.venv/bin/python app_main.py       # 开发
.venv/bin/python build_app.py      # 打包成 .app / .dmg
```

本机起一个只监听 127.0.0.1 的服务，套一个系统 WebKit 窗口。素材**按原路径引用，
一个字节都不复制**。

## 大模型直接驱动

`skills/pipo-highlights` 是一个技能包。桌面版开着的时候它会自动找到本机端口
（写在 `~/Library/Application Support/Pipo/port`），不需要令牌、不联网。

## 它是怎么工作的

```
音频 → 谱通量起音检测 → 候选（召回 0.85，准确 0.44）
     → PANNs CNN14 嵌入 + 逻辑回归重排 → 保留 60%（准确 0.68）
     → 按击球间隔聚成回合 → 按主题排序 → ffmpeg 切片拼接
```

关键在**重排**那一步：检测器的召回已经够了（真击球 85% 在候选里），问题是
一半以上是假货。所以不重做检测，只在候选上做二分类。

## 研究记录

**[docs/RESEARCH.md](docs/RESEARCH.md)** 是从头到现在的完整实验日志，1000 多行，
包括所有失败的尝试和当时的数字。如果你只想看一样东西，看那些「试过并否掉」的章节：

- 高分辨率 log-mel 替代 PANNs——候选级赢了，端到端等于不重排
- MobileNetV3 换骨干——AudioSet mAP 更高、体积小 16 倍，端到端仍然更差
- CNN14 量化——int8 让 45% 的选段变了而且更慢
- 发球四拍模板、交替序列解码、单拍失误检测——都测过，都不成立

一条方法论结论：**候选级 AUC 不是这个产品的有效代理**，四次实验四次给出相反结论。
任何改动都必须跑端到端。

## 标注

`labels/` 里有 8 份人工标注（3434 个候选 / 1895 个真击球）。
`ppai/labels.audit()` 会挡住会静默污染训练集的问题——每条判据都是真出过事之后加的。

## 许可

MIT，见 [LICENSE](LICENSE)。第三方组件见 [THIRD_PARTY.md](THIRD_PARTY.md)
——**ffmpeg 那一条会影响你怎么分发打包好的应用**。
