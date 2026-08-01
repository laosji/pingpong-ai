"""把 Pipo 打包成 macOS .app。

    .venv/bin/python build_app.py

**先做资源校验再打包。** 有两个文件缺了不会报错、只会让产品静默变差：

  * `models/rerank.npz` —— 缺了 `rerank.load()` 返回 None，重排直接失效，
    击球检测准确率从 0.68 掉回 0.38；**而且乒乓球门禁会静默放行一切**
    （没模型时按设计返回 ok=True，那是为了不把用户挡在门外）。
    它在 .gitignore 的 `models/` 里，从干净检出构建时正好会漏。
  * `models/cnn14.onnx` —— 缺了嵌入算不出来，重排同样失效。

这两个都不会崩，只会让成片悄悄变差。所以校验放在最前面，缺就直接退出。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
APP = "Pipo"
VERSION = "0.1.0"

# (路径, 最小字节数) —— 只查存在不够，空文件同样会静默失效
# cnn14.onnx **不在这里** —— 它按需下载（见 ppai/assets.py）：
# 323MB 占安装包七成，而且是冻结的骨干，没理由每次发版重新分发一遍。
# rerank.npz 只有 8KB 但必须随包：它是我们自己训的，每次发版都可能变，
# 而且缺了会让重排静默失效。
REQUIRED = [
    ("models/rerank.npz", 1024),
    ("config.yaml", 512),
    ("web/index.html", 10 * 1024),
    ("bin/ffmpeg", 1024 * 1024),
    ("bin/ffprobe", 1024 * 1024),
    ("app_main.py", 512),
]


def check() -> None:
    bad = []
    for rel, min_size in REQUIRED:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            bad.append("缺少 %s" % rel)
        elif os.path.getsize(p) < min_size:
            bad.append("%s 只有 %d 字节，像是没下全" % (rel, os.path.getsize(p)))
    if bad:
        print("构建中止：")
        for b in bad:
            print("  ✗ %s" % b)
        print("\n  rerank.npz 在 .gitignore 的 models/ 里，从干净检出构建时会漏。")
        print("  用 `python -m ppai.cli rerank-train` 训练后会自动导出。")
        sys.exit(1)
    total = sum(os.path.getsize(os.path.join(ROOT, r)) for r, _ in REQUIRED)
    print("资源校验通过，随包资源合计 %.0f MB\n" % (total / 1e6))


SPEC = r'''# -*- mode: python ; coding: utf-8 -*-
block_cipher = None

a = Analysis(
    ['app_main.py'],
    pathex=[],
    binaries=[('bin/ffmpeg', 'bin'), ('bin/ffprobe', 'bin')],
    datas=[
        ('web', 'web'),
        ('config.yaml', '.'),
        ('models/rerank.npz', 'models'),
        ('ppai', 'ppai'),
        ('server.py', '.'),
    ],
    hiddenimports=[
        # uvicorn 的这几个是运行时按名字加载的，静态分析看不到
        'uvicorn.logging', 'uvicorn.loops.auto', 'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets.auto', 'uvicorn.lifespan.on',
        'onnxruntime', 'onnxruntime.capi._pybind_state',
        # pywebview 的 macOS 后端同样是运行时选的
        'webview.platforms.cocoa',
        'server', 'ppai', 'ppai.paths', 'ppai.media', 'ppai.audio',
        'ppai.embed', 'ppai.rerank', 'ppai.highlight', 'ppai.render',
        'ppai.store', 'ppai.stats', 'ppai.motion', 'ppai.labels',
        'ppai.config', 'ppai.contrib', 'ppai.detect', 'ppai.scene',
    ],
    hookspath=[],
    runtime_hooks=[],
    # 训练和绘图的依赖不进包：torch 529MB、sklearn 47MB、scipy 98MB、
    # matplotlib 33MB，运行时一个都不需要（见 README「去掉推理路径的 torch」）。
    excludes=['torch', 'torchaudio', 'panns_inference', 'sklearn', 'scipy',
              'matplotlib', 'onnx', 'onnxscript', 'torchvision', 'PIL',
              'tkinter', 'test', 'unittest', 'pydoc_data'],
    cipher=block_cipher, noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(pyz, a.scripts, [], exclude_binaries=True,
          name='__APP__', debug=False, bootloader_ignore_signals=False,
          strip=False, upx=False, console=False, target_arch=None,
          codesign_identity=None, entitlements_file=None)

coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas,
               strip=False, upx=False, name='__APP__')

app = BUNDLE(coll, name='__APP__.app', icon=None,
             bundle_identifier='cc.pipo.desktop',
             version='__VERSION__',
             info_plist={
                 'CFBundleName': '__APP__',
                 'CFBundleDisplayName': '__APP__',
                 'CFBundleShortVersionString': '__VERSION__',
                 'NSHighResolutionCapable': True,
                 # 用户的录像常在「下载」「影片」「桌面」里，不声明会被系统挡住
                 'NSDesktopFolderUsageDescription': 'Pipo 需要读取你选择的录像文件',
                 'NSDocumentsFolderUsageDescription': 'Pipo 需要读取你选择的录像文件',
                 'NSDownloadsFolderUsageDescription': 'Pipo 需要读取你选择的录像文件',
                 # 只在本机 127.0.0.1 上跑一个服务，不连外网
                 'NSAppTransportSecurity': {'NSAllowsLocalNetworking': True},
             })
'''


def main() -> None:
    check()
    spec_path = os.path.join(ROOT, "%s.spec" % APP)
    with open(spec_path, "w", encoding="utf-8") as f:
        f.write(SPEC.replace("__APP__", APP).replace("__VERSION__", VERSION))

    for d in ("build", "dist"):
        shutil.rmtree(os.path.join(ROOT, d), ignore_errors=True)

    py = os.path.join(ROOT, ".venv", "bin", "python")
    r = subprocess.run([py, "-m", "PyInstaller", "--noconfirm", "--clean", spec_path],
                       cwd=ROOT)
    if r.returncode != 0:
        sys.exit(r.returncode)

    app_path = os.path.join(ROOT, "dist", "%s.app" % APP)
    if not os.path.isdir(app_path):
        print("构建失败：没有生成 %s" % app_path)
        sys.exit(1)

    size = subprocess.run(["du", "-sm", app_path], capture_output=True, text=True)
    print("\n✓ %s  %s MB" % (app_path, size.stdout.split()[0]))

    # 打完包再查一次：PyInstaller 的 datas 写错路径不会报错，只会少文件
    res = os.path.join(app_path, "Contents", "Resources")
    frm = os.path.join(app_path, "Contents", "Frameworks")
    print("\n包内资源核对：")
    ok = True
    for rel in ("models/rerank.npz", "config.yaml",
                "web/index.html", "bin/ffmpeg", "bin/ffprobe"):
        hit = next((os.path.join(base, rel) for base in (res, frm)
                    if os.path.exists(os.path.join(base, rel))), None)
        if hit:
            print("  ✓ %-22s %.1f MB" % (rel, os.path.getsize(hit) / 1e6))
        else:
            print("  ✗ %s 不在包里" % rel)
            ok = False
    if not ok:
        sys.exit(1)

    dmg = make_dmg()
    print("\n✓ %s  %.0f MB" % (dmg, os.path.getsize(dmg) / 1e6))
    print("\n注意：这个包**没有签名和公证**。用户首次打开会看到「已损坏，"
          "应移到废纸篓」——那是 Gatekeeper 对未签名应用的提示，不是包坏了。"
          "\n  临时绕过：右键点应用 → 打开；或 xattr -dr com.apple.quarantine <app>"
          "\n  正式解决：需要 Apple 开发者账号（99 美元/年）做签名和公证。")




def make_dmg() -> str:
    """打成 .dmg。用 UDZO（zlib 压缩）—— 683MB 的目录压完约 400-500MB。

    带一个 /Applications 的符号链接：不带的话用户会直接在 dmg 里双击运行，
    那是只读磁盘映像，数据目录虽然在 Application Support 不受影响，
    但每次都要挂载映像，而且卸载映像时 app 会被强杀。
    """
    src = os.path.join(ROOT, "dist", "%s.app" % APP)
    stage = os.path.join(ROOT, "dist", "dmg")
    dmg = os.path.join(ROOT, "dist", "%s-%s.dmg" % (APP, VERSION))
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    subprocess.run(["cp", "-R", src, stage], check=True)
    os.symlink("/Applications", os.path.join(stage, "Applications"))
    if os.path.exists(dmg):
        os.remove(dmg)
    subprocess.run(["hdiutil", "create", "-volname", APP, "-srcfolder", stage,
                    "-ov", "-format", "UDZO", dmg], check=True)
    shutil.rmtree(stage, ignore_errors=True)
    return dmg


if __name__ == "__main__":
    main()
