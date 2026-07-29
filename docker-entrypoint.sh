#!/bin/sh
set -e
# PANNs 权重 312MB，不打进镜像 —— 打进去每次重建都要重传，
# 放挂卷里下一次，之后所有重建都复用。
W="$HOME/panns_data"
mkdir -p "$W"
if [ ! -s "$W/Cnn14_mAP=0.431.pth" ]; then
  echo "[init] 下载 PANNs 权重（312MB，只需一次）…"
  curl -fsSL --retry 3 -o "$W/Cnn14_mAP=0.431.pth" \
    "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
fi
if [ ! -s "$W/class_labels_indices.csv" ]; then
  curl -fsSL --retry 3 -o "$W/class_labels_indices.csv" \
    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
fi
# 可变数据全部落在挂卷上；软链回代码目录，省得改一堆路径常量
for d in labels models uploads out cache; do
  mkdir -p "$PIPO_DATA/$d"
  [ -L "/app/$d" ] || ln -sfn "$PIPO_DATA/$d" "/app/$d"
done
exec "$@"
