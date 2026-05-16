cd /root/autodl-tmp/cv-project

# ① 一把全跑（small + middle，全部 4 个诊断任务）
bash scripts/run_diagnostics.sh

# ② 只跑某个任务（例如只画 per-epoch 曲线）
bash scripts/run_diagnostics.sh --task per_epoch

# ③ 只对某个模型
bash scripts/run_diagnostics.sh --model small
bash scripts/run_diagnostics.sh --model middle

# ④ 任务 + 模型组合（最常用）
bash scripts/run_diagnostics.sh --task pca       --model small
bash scripts/run_diagnostics.sh --task features  --model both
bash scripts/run_diagnostics.sh --task layer4    --model middle

# ⑤ 指定 PCA 用哪张图
bash scripts/run_diagnostics.sh --task pca --image datasets/test/navi_resized/<某张>.jpg

# ⑥ 改 epoch tag（默认 014，如果你最后一个评估点不是 epoch14）
bash scripts/run_diagnostics.sh --task layer4 --epoch_tag 011

# ⑦ 改诊断采样图数（默认 30）
bash scripts/run_diagnostics.sh --task features --num_images 50