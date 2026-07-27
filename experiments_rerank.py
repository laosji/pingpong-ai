import json, sys
import numpy as np
sys.path.insert(0,"/Users/duanchao/Downloads/pingpong-ai")
from ppai import audio, config, embed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
CACHE="/private/tmp/claude-501/-Users-duanchao-Downloads/f577a54d-6089-4994-be12-45d55cc86df8/scratchpad/cache"
VID="/Users/duanchao/Downloads/PP-video/06a6c98c76ebd0a720a75f2f3ab0d352.mp4"
cfg=config.load()
pcm=audio.extract_pcm(VID,cfg["audio"]["sr"])
det,env,thr,fr=audio.detect_hits(pcm,cfg["audio"])
amps=audio.hit_amplitudes(det,env,fr)
L=json.load(open("labels/06a6c98c76ebd0a720a75f2f3ab0d352.json"))
A,B=L["complete_ranges"][0]
truth=np.array(sorted([(a+b)/2 for a,b in L["playing"] if A<=(a+b)/2<=B]))
m=(det>=A)&(det<=B); C=det[m]; CA=amps[m]
y=np.array([1 if np.any(np.abs(truth-t)<=0.35) else 0 for t in C])
print("窗内候选 %d 个：真击球 %d / 误报 %d（当前准确率 %.3f）\n"%(len(C),y.sum(),(1-y).sum(),y.mean()))

# 特征1：手工 —— 强度、局部密度、与相邻候选的间隔
def handfeat(i):
    t=C[i]
    near=C[(C>=t-1.5)&(C<=t+1.5)]
    prev=C[i-1] if i>0 else t-9
    nxt =C[i+1] if i<len(C)-1 else t+9
    return [CA[i], np.log1p(len(near)), min(t-prev,5), min(nxt-t,5),
            CA[i]/max(np.median(CA[(C>=t-3)&(C<=t+3)]),1e-6)]
F1=np.array([handfeat(i) for i in range(len(C))])
# 特征2：PANNs 嵌入（候选时刻的 1 秒窗）
t2,f2=embed.embed_video(VID,cache_dir=CACHE)
idx=np.array([np.argmin(np.abs(t2-t)) for t in C])
F2=f2[idx]
tr=C<180; te=C>=180
print("训练 120-180s(%d候选,%d真) | 测试 180-240s(%d候选,%d真)\n"%(tr.sum(),y[tr].sum(),te.sum(),y[te].sum()))
def run(X,name):
    c=LogisticRegression(max_iter=5000,C=0.5,class_weight="balanced").fit(X[tr],y[tr])
    p=c.predict_proba(X[te])[:,1]
    a=roc_auc_score(y[te],p)
    line=""
    for k in [0.5,0.6,0.7]:
        sel=p>k
        pr=y[te][sel].mean() if sel.sum() else float("nan")
        rc=y[te][sel].sum()/max(y[te].sum(),1)
        line+="  阈值%.1f→准确%.3f 召回%.3f"%(k,pr,rc)
    print("%-22s AUC %.3f%s"%(name,a,line))
run(F1,"手工特征")
run(F2,"PANNs 嵌入")
run(np.hstack([F1,F2]),"手工+PANNs")
print("\n（基线：不重排时准确率 %.3f，召回 1.000）"%y[te].mean())
