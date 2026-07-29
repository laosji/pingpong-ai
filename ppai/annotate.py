"""生成自包含的标注页面（对应方案「模块八：人工标注学习系统」的标注界面）。

设计取舍：
  * 带视频播放器 —— 判断「有没有在打球」看画面远快于听声音，声谱图只作时间轴参考。
  * 视频用相对路径引用而非内嵌 base64 —— 一小时录像内嵌会产生几百 MB 的 HTML。
    代价是 HTML 必须和视频保持相对位置不变。
  * 可选叠加 AI 预测区间，人工只需修正 —— 这正是方案里的数据闭环入口。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import audio as A

_HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>标注 - __NAME__</title>
<style>
 body{margin:0;font:14px/1.5 -apple-system,"PingFang SC",sans-serif;background:#16181d;color:#e6e6e6}
 header{padding:10px 16px;background:#1e2128;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 header b{font-weight:600}
 video{max-height:46vh;background:#000;display:block;margin:0 auto}
 #wrap{overflow-x:auto;overflow-y:hidden;border-top:1px solid #2c3038;position:relative}
 #inner{position:relative;height:__H__px}
 #spec{position:absolute;top:0;left:0;height:__H__px;image-rendering:pixelated}
 #cv{position:absolute;top:0;left:0}
 button{background:#2c3038;color:#e6e6e6;border:1px solid #3a3f4a;border-radius:5px;
        padding:5px 11px;cursor:pointer;font-size:13px}
 button:hover{background:#3a3f4a}
 .k{color:#8b93a1;font-size:12px}
 kbd{background:#2c3038;border:1px solid #444;border-radius:3px;padding:1px 5px;font-size:11px}
 #stat{margin-left:auto;color:#9fb3c8}
</style>
<header>
  <b>__NAME__</b>
  <button onclick="tog()">播放/暂停 <kbd>空格</kbd></button>
  <button id="modebtn" onclick="toggleMode()">模式: 击球 <kbd>M</kbd> 切换</button>
  <button onclick="delSel()">删除选中 <kbd>Del</kbd></button>
  <button onclick="addServe()">标发球 <kbd>S</kbd></button>
  <button onclick="undoServe()">撤销发球 <kbd>Shift+S</kbd></button>
  <button onclick="setComplete()">标记为「已标完」</button>
  <button onclick="dl()">导出 JSON</button>
  <span id="stat"></span>
</header>
<video id="v" src="__VIDEO__" controls preload="metadata"></video>
<div id="wrap"><div id="inner">
  <img id="spec" src="data:image/png;base64,__SPEC__">
  <canvas id="cv" width="__W__" height="__H__"></canvas>
</div></div>
<p class="k" style="padding:8px 16px">
  __WINNOTE__<br>
  时间轴上 <b>拖动</b> 标一次击球 · <b>单击</b> 定位 · <b>点已标记</b> 选中 · <kbd>Del</kbd> 删除 ·
  <kbd>←</kbd><kbd>→</kbd> 0.2 秒微调（按住 Shift 为 5 秒）
  <span style="color:#6f7684">　<span style="color:#3ecf80">绿=击球</span>　<span style="color:#e8933a">橙=捡球</span>　蓝=AI 预测</span>
</p>
<script>
const DUR=__DUR__, PPS=__PPS__, W=__W__, H=__H__, T0=__T0__, T1=__T1__;
let lab=__LABEL__, pred=__PRED__, sel=-1, drag=null;
if(!lab.pickup) lab.pickup=[];
if(!lab.serves) lab.serves=[];
// 两类标注分开存：捡球是负例，混进 playing 会让真值失效
let mode='playing';   // 'playing' | 'pickup'
const COLOR={playing:['rgba(80,220,140,.22)','rgba(80,220,140,.42)','#3ecf80','#7dffb0'],
             pickup :['rgba(255,170,60,.22)','rgba(255,170,60,.42)','#e8933a','#ffc078']};
const v=document.getElementById('v'), cv=document.getElementById('cv'),
      cx=cv.getContext('2d'), wrap=document.getElementById('wrap');
// 窗口模式：时间轴只覆盖 [T0,T1]，坐标要带偏移
const x2t=x=>Math.max(T0,Math.min(T1,T0+x/PPS)), t2x=t=>(t-T0)*PPS;
// 时刻换算带偏移，时长换算不能带 —— 别拿 t2x 去算宽度
const d2x=d=>d*PPS;

function draw(){
  cx.clearRect(0,0,W,H);
  // AI 预测：顶部细条，仅参考
  cx.fillStyle='rgba(90,150,255,.30)';
  pred.forEach(s=>cx.fillRect(t2x(s[0]),0,d2x(s[1]-s[0]),9));
  // 人工标注：两类各自颜色，当前模式的那类才能选中
  ['pickup','playing'].forEach(kind=>{
    const c=COLOR[kind];
    lab[kind].forEach((s,i)=>{
      const on = (kind===mode && i===sel);
      cx.fillStyle = on?c[1]:c[0];
      cx.fillRect(t2x(s[0]),10,Math.max(2,d2x(s[1]-s[0])),H-10);
      cx.strokeStyle = on?c[3]:c[2]; cx.lineWidth=on?2:1;
      cx.strokeRect(t2x(s[0]),10,Math.max(2,d2x(s[1]-s[0])),H-10);
    });
  });
  cx.strokeStyle='rgba(255,180,60,.85)'; cx.lineWidth=1;
  lab.hits.forEach(t=>{cx.beginPath();cx.moveTo(t2x(t),H-22);cx.lineTo(t2x(t),H);cx.stroke();});
  // 发球画整条竖线 + 顶部三角，和击球的短刻度明显区分 ——
  // 它标的是「回合从这里开始」，看的是位置对不对，得贯穿整个时间轴
  cx.strokeStyle='#5ac8ff'; cx.lineWidth=1.5; cx.fillStyle='#5ac8ff';
  lab.serves.forEach(t=>{const x=t2x(t);
    cx.beginPath();cx.moveTo(x,0);cx.lineTo(x,H);cx.stroke();
    cx.beginPath();cx.moveTo(x-5,0);cx.lineTo(x+5,0);cx.lineTo(x,9);cx.closePath();cx.fill();});
  if(drag){ cx.fillStyle='rgba(80,220,140,.25)';
    cx.fillRect(t2x(Math.min(drag.a,drag.b)),10,d2x(Math.abs(drag.b-drag.a)),H-10); }
  const px=t2x(v.currentTime);
  cx.strokeStyle='#ff5c5c'; cx.lineWidth=2;
  cx.beginPath();cx.moveTo(px,0);cx.lineTo(px,H);cx.stroke();
  const inWin=lab.playing.filter(s=>s[1]>T0&&s[0]<T1).length;
  const puWin=lab.pickup.filter(s=>s[1]>T0&&s[0]<T1).length;
  const puSec=lab.pickup.filter(s=>s[1]>T0&&s[0]<T1).reduce((a,s)=>a+s[1]-s[0],0);
  const mmss=x=>`${Math.floor(x/60)}:${String(Math.floor(x%60)).padStart(2,'0')}`;
  const svWin=lab.serves.filter(t=>t>T0&&t<T1).length;
  document.getElementById('stat').textContent =
    `${mmss(v.currentTime)} · 窗口 ${mmss(T0)}–${mmss(T1)} · 发球 ${svWin} 次 · 击球 ${inWin} 次 · 捡球 ${puWin} 段/${puSec.toFixed(0)}秒`;
}
function tick(){draw();
  const px=t2x(v.currentTime);
  if(px<wrap.scrollLeft||px>wrap.scrollLeft+wrap.clientWidth-60)
    wrap.scrollLeft=px-wrap.clientWidth*0.35;
  // 窗口模式下播出界就停，避免不知不觉标到窗口外
  if(v.currentTime>T1+0.2&&!v.paused) v.pause();
  requestAnimationFrame(tick);}
function toggleMode(){ mode = mode==='playing'?'pickup':'playing'; sel=-1;
  document.getElementById('modebtn').innerHTML =
    '模式: '+(mode==='playing'?'击球':'<span style="color:#ffc078">捡球</span>')+' <kbd>M</kbd> 切换';
  draw(); }
function norm(){ // 排序 + 合并重叠，和 labels.py 保持一致
  ['playing','pickup'].forEach(k=>{ lab[k]=lab[k].map(s=>[Math.max(0,Math.min(...s)),Math.min(DUR,Math.max(...s))])
    .filter(s=>s[1]-s[0]>0.05).sort((a,b)=>a[0]-b[0])
    .reduce((o,s)=>{const l=o[o.length-1];
      if(l&&s[0]<=l[1]) l[1]=Math.max(l[1],s[1]); else o.push(s); return o;},[]); });
}
cv.onmousedown=e=>{const t=x2t(e.offsetX);
  const hit=lab[mode].findIndex(s=>t>=s[0]&&t<=s[1]);
  if(hit>=0&&!e.shiftKey){sel=hit;v.currentTime=t;draw();return;}
  drag={a:t,b:t};};
cv.onmousemove=e=>{if(drag){drag.b=x2t(e.offsetX);draw();}};
cv.onmouseup=e=>{if(!drag)return;
  const a=Math.min(drag.a,drag.b),b=Math.max(drag.a,drag.b);
  if(b-a<0.08){v.currentTime=a;sel=-1;} else {lab[mode].push([a,b]);norm();}
  drag=null;draw();};
let mk=null;
function mark(){ if(mk===null){mk=v.currentTime;} else {
    lab.playing.push([Math.min(mk,v.currentTime),Math.max(mk,v.currentTime)]);mk=null;norm();} draw();}
function tog(){v.paused?v.play():v.pause();}
function delSel(){if(sel>=0){lab[mode].splice(sel,1);sel=-1;draw();}}
// 发球只记时间点，不记区间 —— 回合起点是一个时刻，不是一段
function addServe(){lab.serves.push(v.currentTime);
  lab.serves.sort((a,b)=>a-b);draw();}
function undoServe(){
  // 撤掉离当前播放头最近的那个，而不是最后加的 ——
  // 标错时人是回到出错的位置去改，不是回到时间顺序的末尾
  if(!lab.serves.length) return;
  let bi=0,bd=1e9;
  lab.serves.forEach((t,i)=>{const d=Math.abs(t-v.currentTime); if(d<bd){bd=d;bi=i;}});
  lab.serves.splice(bi,1);draw();}
function setComplete(){lab.complete=!lab.complete;draw();}
function dl(){norm();
  ['playing','pickup'].forEach(k=>lab[k]=lab[k].map(s=>[+s[0].toFixed(3),+s[1].toFixed(3)]));
  lab.hits=lab.hits.map(t=>+t.toFixed(3)).sort((a,b)=>a-b);
  lab.serves=lab.serves.map(t=>+t.toFixed(3)).sort((a,b)=>a-b);
  const b=new Blob([JSON.stringify(lab,null,2)],{type:'application/json'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(b);a.download='__STEM__.json';a.click();}
// 不支持 Range 请求的服务器（如 python -m http.server）会让视频无法跳转，
// 而跳转是这个工具的全部前提。静默失败很难查，所以显式报警。
function seekCheck(){
  // seekable 可能比 loadeddata 稍晚才填好，给一点余量
  setTimeout(()=>{
  const ok = v.seekable.length && v.seekable.end(0) > 0.5;
  if(!ok && !document.getElementById('seekwarn')){
    const b=document.createElement('div');
    b.id='seekwarn';
    b.style.cssText='background:#7a2b2b;color:#ffdede;padding:8px 16px;font-size:13px';
    b.textContent='⚠ 当前来源不支持跳转（服务器缺少 HTTP Range 支持），时间轴定位会失效。'
      +'请直接双击 HTML 用 file:// 打开。';
    document.body.insertBefore(b, document.body.firstChild);
  }}, 500);
}
// 脚本执行时视频可能已加载完毕，loadeddata 早已错过 —— 两条路都要走
if(v.readyState>=2) seekCheck(); else v.addEventListener('loadeddata',seekCheck,{once:true});
document.onkeydown=e=>{
  if(e.target.tagName==='INPUT')return;
  const k=e.key.toLowerCase();
  if(e.code==='Space'){e.preventDefault();tog();}
  else if(k==='i'){mk=v.currentTime;draw();}
  else if(k==='o'&&mk!==null){lab[mode].push([Math.min(mk,v.currentTime),
      Math.max(mk,v.currentTime)]);mk=null;norm();draw();}
  else if(k==='f'){lab.hits.push(v.currentTime);draw();}
  else if(k==='s'&&e.shiftKey){e.preventDefault();undoServe();}
  else if(k==='s'){e.preventDefault();addServe();}
  else if(k==='m'){toggleMode();}
  else if(e.key==='Delete'||e.key==='Backspace'){e.preventDefault();delSel();}
  else if(e.key==='ArrowLeft'){e.preventDefault();v.currentTime=Math.max(T0,v.currentTime-(e.shiftKey?5:0.2));}
  else if(e.key==='ArrowRight'){e.preventDefault();v.currentTime=Math.min(T1,v.currentTime+(e.shiftKey?5:0.2));}
};
// 窗口模式：一进来就定位到窗口开头
function goStart(){ if(v.currentTime<T0||v.currentTime>T1) v.currentTime=T0; }
if(v.readyState>=1) goStart(); else v.addEventListener('loadedmetadata',goStart,{once:true});
tick();
</script>
"""


def _spectrogram_png(pcm: np.ndarray, sr: int, width: int, height: int) -> str:
    import base64
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = 100.0
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.specgram(pcm, NFFT=512, Fs=sr, noverlap=256, cmap="magma", vmin=-120, vmax=-30)
    ax.set_ylim(0, 8000)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build(video: str, duration: float, out_html: str, label: Dict,
          pred_segments: Optional[List[Dict]] = None, sr: int = 16000,
          height: int = 150, max_width: int = 28000,
          window: Optional[Tuple[float, float]] = None) -> str:
    """生成标注页面。max_width 是浏览器 canvas 的安全上限。

    window=(t0,t1) 时只标注该时段：分辨率可以拉得很高，且自动把该时段
    写入 complete_ranges —— 声明「这段我逐个标完了」，评测才敢在其中算准确率。
    """
    t0, t1 = window if window else (0.0, duration)
    t0 = max(0.0, min(t0, duration))
    t1 = max(t0 + 1.0, min(t1, duration))
    span = t1 - t0

    # 窗口越短，每秒像素越多；2 分钟的窗可以到 200px/s，即每像素 5 毫秒
    pps = max(4.0, min(200.0, max_width / span))
    width = max(1, int(span * pps))

    pcm = A.extract_pcm(video, sr)
    if window:
        pcm = pcm[int(t0 * sr):int(t1 * sr)]
        label = dict(label)
        # 反复对同一窗口生成页面时会累积重复项，合并重叠后再写回
        merged = list(label.get("complete_ranges") or []) + [[round(t0, 3), round(t1, 3)]]
        out: List[List[float]] = []
        for a, b in sorted(merged):
            if out and a <= out[-1][1]:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        label["complete_ranges"] = out
    spec = _spectrogram_png(pcm, sr, width, height) if len(pcm) > 512 else ""

    rel = os.path.relpath(os.path.abspath(video), os.path.dirname(os.path.abspath(out_html)))
    stem = os.path.splitext(os.path.basename(video))[0]
    pred = [[s["start"], s["end"]] for s in (pred_segments or [])]

    def mmss(x):
        return "%d:%02d" % (int(x) // 60, int(x) % 60)

    if window:
        note = ("<b style='color:#7dffb0'>穷尽标注窗口 %s – %s（%.0f 秒）</b>："
                "请把这段里<b>每一次</b>击球都标上，漏一个都会被算成误报。"
                "导出的 JSON 会自动带上 complete_ranges，评测只在此区间算准确率。"
                % (mmss(t0), mmss(t1), span))
    else:
        note = "整片模式：未声明 complete_ranges，导出的标注只能算召回，不能算准确率。"

    html = _HTML
    for k, val in (("__NAME__", os.path.basename(video)),
                   ("__VIDEO__", rel.replace("\\", "/")),
                   ("__SPEC__", spec),
                   ("__STEM__", stem.replace("'", "")),
                   ("__DUR__", "%.3f" % duration),
                   ("__PPS__", "%.4f" % pps),
                   ("__W__", str(width)),
                   ("__H__", str(height)),
                   ("__T0__", "%.3f" % t0),
                   ("__T1__", "%.3f" % t1),
                   ("__WINNOTE__", note),
                   ("__LABEL__", json.dumps(label, ensure_ascii=False)),
                   ("__PRED__", json.dumps(pred))):
        html = html.replace(k, val)

    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html
