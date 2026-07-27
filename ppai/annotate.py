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
from typing import Dict, List, Optional

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
  <button onclick="mark()">标记打球段 <kbd>I</kbd>起 <kbd>O</kbd>止</button>
  <button onclick="delSel()">删除选中 <kbd>Del</kbd></button>
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
  时间轴上 <b>拖动</b> 新建打球区间 · <b>单击</b> 定位 · <b>点区间</b> 选中 · 拖区间两端调边界 ·
  <kbd>←</kbd><kbd>→</kbd> 逐秒 · <kbd>F</kbd> 标记一次击球
  <span style="color:#6f7684">　绿=人工标注　蓝=AI 预测（仅参考）</span>
</p>
<script>
const DUR=__DUR__, PPS=__PPS__, W=__W__, H=__H__;
let lab=__LABEL__, pred=__PRED__, sel=-1, drag=null;
const v=document.getElementById('v'), cv=document.getElementById('cv'),
      cx=cv.getContext('2d'), wrap=document.getElementById('wrap');
const x2t=x=>Math.max(0,Math.min(DUR,x/PPS)), t2x=t=>t*PPS;

function draw(){
  cx.clearRect(0,0,W,H);
  // AI 预测：顶部细条，仅参考
  cx.fillStyle='rgba(90,150,255,.30)';
  pred.forEach(s=>cx.fillRect(t2x(s[0]),0,t2x(s[1]-s[0]),9));
  // 人工标注
  lab.playing.forEach((s,i)=>{
    cx.fillStyle = i===sel?'rgba(80,220,140,.42)':'rgba(80,220,140,.22)';
    cx.fillRect(t2x(s[0]),10,t2x(s[1]-s[0]),H-10);
    cx.strokeStyle = i===sel?'#7dffb0':'#3ecf80'; cx.lineWidth=i===sel?2:1;
    cx.strokeRect(t2x(s[0]),10,t2x(s[1]-s[0]),H-10);
  });
  cx.strokeStyle='rgba(255,180,60,.85)'; cx.lineWidth=1;
  lab.hits.forEach(t=>{cx.beginPath();cx.moveTo(t2x(t),H-22);cx.lineTo(t2x(t),H);cx.stroke();});
  if(drag){ cx.fillStyle='rgba(80,220,140,.25)';
    cx.fillRect(t2x(Math.min(drag.a,drag.b)),10,t2x(Math.abs(drag.b-drag.a)),H-10); }
  const px=t2x(v.currentTime);
  cx.strokeStyle='#ff5c5c'; cx.lineWidth=2;
  cx.beginPath();cx.moveTo(px,0);cx.lineTo(px,H);cx.stroke();
  const tot=lab.playing.reduce((a,s)=>a+s[1]-s[0],0);
  document.getElementById('stat').textContent =
    `${v.currentTime.toFixed(1)}s / ${DUR.toFixed(1)}s · 已标 ${lab.playing.length} 段 `+
    `共 ${tot.toFixed(1)}s (${(100*tot/DUR).toFixed(0)}%) · ${lab.complete?'已标完':'未标完'}`;
}
function tick(){draw();
  const px=t2x(v.currentTime);
  if(px<wrap.scrollLeft||px>wrap.scrollLeft+wrap.clientWidth-60)
    wrap.scrollLeft=px-wrap.clientWidth*0.35;
  requestAnimationFrame(tick);}
function norm(){ // 排序 + 合并重叠，和 labels.py 保持一致
  lab.playing=lab.playing.map(s=>[Math.max(0,Math.min(...s)),Math.min(DUR,Math.max(...s))])
    .filter(s=>s[1]-s[0]>0.05).sort((a,b)=>a[0]-b[0])
    .reduce((o,s)=>{const l=o[o.length-1];
      if(l&&s[0]<=l[1]) l[1]=Math.max(l[1],s[1]); else o.push(s); return o;},[]);
}
cv.onmousedown=e=>{const t=x2t(e.offsetX);
  const hit=lab.playing.findIndex(s=>t>=s[0]&&t<=s[1]);
  if(hit>=0&&!e.shiftKey){sel=hit;v.currentTime=t;draw();return;}
  drag={a:t,b:t};};
cv.onmousemove=e=>{if(drag){drag.b=x2t(e.offsetX);draw();}};
cv.onmouseup=e=>{if(!drag)return;
  const a=Math.min(drag.a,drag.b),b=Math.max(drag.a,drag.b);
  if(b-a<0.08){v.currentTime=a;sel=-1;} else {lab.playing.push([a,b]);norm();}
  drag=null;draw();};
let mk=null;
function mark(){ if(mk===null){mk=v.currentTime;} else {
    lab.playing.push([Math.min(mk,v.currentTime),Math.max(mk,v.currentTime)]);mk=null;norm();} draw();}
function tog(){v.paused?v.play():v.pause();}
function delSel(){if(sel>=0){lab.playing.splice(sel,1);sel=-1;draw();}}
function setComplete(){lab.complete=!lab.complete;draw();}
function dl(){norm();
  lab.playing=lab.playing.map(s=>[+s[0].toFixed(3),+s[1].toFixed(3)]);
  lab.hits=lab.hits.map(t=>+t.toFixed(3)).sort((a,b)=>a-b);
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
  else if(k==='o'&&mk!==null){lab.playing.push([Math.min(mk,v.currentTime),
      Math.max(mk,v.currentTime)]);mk=null;norm();draw();}
  else if(k==='f'){lab.hits.push(v.currentTime);draw();}
  else if(e.key==='Delete'||e.key==='Backspace'){e.preventDefault();delSel();}
  else if(e.key==='ArrowLeft'){e.preventDefault();v.currentTime=Math.max(0,v.currentTime-(e.shiftKey?5:1));}
  else if(e.key==='ArrowRight'){e.preventDefault();v.currentTime=Math.min(DUR,v.currentTime+(e.shiftKey?5:1));}
};
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
          height: int = 150, max_width: int = 28000) -> str:
    """生成标注页面。max_width 是浏览器 canvas 的安全上限。"""
    pps = min(100.0, max_width / max(duration, 1e-6))
    pps = max(pps, 4.0)
    width = max(1, int(duration * pps))

    pcm = A.extract_pcm(video, sr)
    spec = _spectrogram_png(pcm, sr, width, height) if len(pcm) > 512 else ""

    rel = os.path.relpath(os.path.abspath(video), os.path.dirname(os.path.abspath(out_html)))
    stem = os.path.splitext(os.path.basename(video))[0]
    pred = [[s["start"], s["end"]] for s in (pred_segments or [])]

    html = _HTML
    for k, val in (("__NAME__", os.path.basename(video)),
                   ("__VIDEO__", rel.replace("\\", "/")),
                   ("__SPEC__", spec),
                   ("__STEM__", stem.replace("'", "")),
                   ("__DUR__", "%.3f" % duration),
                   ("__PPS__", "%.4f" % pps),
                   ("__W__", str(width)),
                   ("__H__", str(height)),
                   ("__LABEL__", json.dumps(label, ensure_ascii=False)),
                   ("__PRED__", json.dumps(pred))):
        html = html.replace(k, val)

    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html
