# Paper Reading Guide — DINOv3 + LoRA + HardInfoNCE + Safe Radius

> 面向**第三者**的快速上手手册：30 分钟读完即可独立看懂 [`main_refine.tex`](main_refine.tex)、跑通仓库、复现表格里的每一个数字。

---

## 0. TL;DR（你不读论文也要知道的 5 句话）

1. **目标**：在不动 DINOv3 任何参数的前提下，让它的 patch 特征更适合 NAVI 的稠密匹配 + 位姿估计。
2. **手段**：LoRA(rank 4, α=8) 只插到每个 Transformer block 的 **Q、V** 投影上 + **HardInfoNCE** 损失 + **5-patch Safe-Radius** 邻域屏蔽。
3. **数据**：单物体多视角数据集 **NAVI**，5 596 训练 pair / 908 测试 pair；用深度回投得到稠密对应作为正样本。
4. **结果**：ViT-S/16 与 ViT-L/16 上 Precision 都 +7pp 左右，AUC@20 翻倍；**但 AUC@5 不涨**，PCA / 表征统计几乎不变 → 我们诚实地承认这只是一次"低秩小幅旋转"，不是表征级重塑。
5. **复现**：`bash scripts/train_oneclick.sh`（S/16）+ `bash scripts/run_diagnostics.sh`（出图）→ 对照 [main_refine.tex](main_refine.tex) 的图表即可。

---

## 1. 论文叙事一图流

```
                ┌───────────────────────┐
                │ DINOv3 (frozen)       │  ViT-S/16 或 ViT-L/16
                └──────────┬────────────┘
                           │ patch tokens
        ┌──────────────────┴──────────────────┐
        │ LoRA r=4, α=8  仅 Q, V              │   ← 唯一可训练的 0.x% 参数
        └──────────────────┬──────────────────┘
                           │
   NAVI pair (A,B) ──► 深度回投取正样本 ──► HardInfoNCE
                                              │
                              Safe-Radius=5 邻域 mask（避免把同一物体表面拉开）
                                              │
                            ┌─────────────────┴─────────────────┐
                            │ 评估: MNN → USAC → 5-pt pose       │
                            │ 指标: AUC@5/10/20, Precision       │
                            └────────────────────────────────────┘
```

---

## 2. 论文章节速览（与代码/数据的双向索引）

| 论文位置 | 主要内容 | 对应代码/产物 |
|---|---|---|
| §2 Background | DINOv3 / 稠密匹配 / NAVI 评估流程 | [`finetune/extract_and_match_hf.py`](finetune/extract_and_match_hf.py)、[`evaluate/evaluate_csv_essential.py`](evaluate/evaluate_csv_essential.py) |
| §3 Method | LoRA 注入位置 + HardInfoNCE + Safe Radius | [`finetune/lora_hf.py`](finetune/lora_hf.py)、[`finetune/loss.py`](finetune/loss.py) |
| §3 Training Recipe | AdamW lr=1e-4, wd=1e-4, cosine, 15 epoch, batch=8 | [`finetune/train_lora_hf.py`](finetune/train_lora_hf.py)、[`scripts/train_oneclick.sh`](scripts/train_oneclick.sh) |
| §4.1 Per-Epoch Dynamics | 两个 backbone 的 epoch 曲线 | [`presentation/result/per_epoch_small_vs_middle.png`](presentation/result/per_epoch_small_vs_middle.png) ← [`presentation/plot_per_epoch.py`](presentation/plot_per_epoch.py) |
| §4.2 Final Numerical Results | 最终表格 | [`output/navi_small/eval_per_epoch/summary.tsv`](output/navi_small/eval_per_epoch/summary.tsv)、[`output/navi_middle/eval_per_epoch/summary.tsv`](output/navi_middle/eval_per_epoch/summary.tsv) |
| §4.4 Diagnostic Analysis | hist_intra / hist_pos / bars / PCA / Layer4 | [`presentation/diagnostics_features.py`](presentation/diagnostics_features.py)、[`presentation/diagnostics_layer4.py`](presentation/diagnostics_layer4.py)、[`presentation/pca_visualizer.py`](presentation/pca_visualizer.py)，输出在 [`presentation/result/diag_small/`](presentation/result/diag_small) 与 [`presentation/result/diag_middle/`](presentation/result/diag_middle) |
| §4.5 Limitations | AUC@5 不涨 / 表征不动 / 单物体 / 未对比专用 matcher | 与 §4.4 的图一一对应 |
| §5 Conclusion | 诚实定性："a small low-rank tilt" | — |

---

## 3. 核心公式与代码的一一对应

### 3.1 LoRA（[`finetune/lora_hf.py`](finetune/lora_hf.py)）

数学：`y = Wx + (α/r) · B (A x)`，其中 `A ∈ R^{r×d}` 高斯初始化，`B ∈ R^{d×r}` **零初始化**。

代码要点：
- 仅注入 `attention.q_proj` 与 `attention.v_proj`；K、O、FFN 全部冻结。
- 通过 `register_module` / `forward` 拦截，原 `q_proj` 不动；`B=0` 保证 step 0 输出与 zero-shot 完全一致。
- 这一性质在论文 §3 与 §4.3 Discussion 第 1 条被反复用到：训练只能"加信息"，不会先把已有的好特征打乱。

### 3.2 HardInfoNCE + Safe Radius（[`finetune/loss.py`](finetune/loss.py)）

```
L = -log [ exp(<a,p>/τ) / ( exp(<a,p>/τ) + Σ_{n∈HardK} exp(<a,n>/τ) ) ]
```

- `τ = 0.07`，`K = 128` 个最难负样本。
- **Safe Radius = 5**：对 anchor 在自身图内的负样本，凡 2D patch grid 距离 ≤ 5 的全部 mask 掉——这避免把"同一物理表面上的相邻 patch"当负样本对抗，从而保住 DINOv3 自带的局部光滑性先验。
- `batch_size = 8` 意味着每个 anchor 还会从**另外 7 张不同场景图**里取负样本（真正的跨图负样本），避免坍塌到 `ln(K+1)` 的平凡下界。

### 3.3 正样本生成（[`finetune/dataset.py`](finetune/dataset.py) + [`finetune/generate_train_pairs.py`](finetune/generate_train_pairs.py)）

NAVI 自带深度图 + 相机位姿 → 把 A 的每个 patch 中心回投到 B → 互投影误差 < 阈值的视为正对应 → 保留为训练正样本。

---

## 4. 怎样从 0 复现论文里的每一个数字

```bash
# 0. 环境
bash scripts/setup_env.sh

# 1. 训练 + 每 3 epoch 自动评估 + 汇总（任一 backbone 即可）
bash scripts/train_oneclick.sh           # ViT-S/16
bash scripts/train_oneclick_middle.sh    # ViT-L/16
# → output/navi_{small,middle}/eval_per_epoch/summary.tsv 即论文 Table（§4.2）

# 2. per-epoch 曲线（论文 Fig 1 / §4.1）
python presentation/plot_per_epoch.py

# 3. 全部诊断图（论文 §4.4 全部子图）
bash scripts/run_diagnostics.sh
# 单独项目用法见 analysis.sh
```

最佳 epoch 取规则与论文一致：在 `summary.tsv` 中按 **Precision** 取最大行；S/16 落在 epoch 12，L/16 落在 epoch 14。

---

## 5. 论文里"诚实承认的不足"——一定要看的部分

§4.5 与 §5 末段都明确写了，第三者最容易忽视，但这是本论文最有信号的一节：

1. **AUC@5 不提升**：S/16 从 0.076 → 0.000，L/16 从 0.059 → 0.061。也就是说严格阈值下的位姿正确率没有被改善，"+7pp Precision" 主要来自更宽阈值的边缘成功对。
2. **表征几乎没动**：mean intra-cos / pos-cos / 有效秩 / 邻域占优 / PCA-RGB 在 LoRA 前后差异 ≤ 0.02；PCA 视觉上几乎肉眼无差。
3. **Layer-4 不是 pose-disentangled**：中间层与相机位姿角度的 cosine 关系仍然弱，LoRA 没有选择性地重塑几何相关层。
4. **只在单物体 NAVI 上验证**，多场景（MegaDepth/ScanNet/IMC）未测。
5. **没和 SuperPoint / LoFTR / DINOv2-with-decoder** 等专用 matcher 对比。

读到这里你就能正确定位本工作的边界：**它是"在冻结 DINOv3 上加一个对 NAVI 决策面友好的小幅低秩旋转"，不是一个新表征**。

---

## 6. 看代码的推荐顺序（30 分钟版）

1. [`finetune/lora_hf.py`](finetune/lora_hf.py) — 看 LoRA 是怎么挂到 Q/V 的（≈ 5 min）。
2. [`finetune/loss.py`](finetune/loss.py) — Safe-Radius mask + HardInfoNCE（≈ 5 min）。
3. [`finetune/train_lora_hf.py`](finetune/train_lora_hf.py) — 训练循环、每 3 epoch 评估钩子（≈ 5 min）。
4. [`finetune/extract_and_match_hf.py`](finetune/extract_and_match_hf.py) — 推理与 MNN 匹配（≈ 5 min）。
5. [`evaluate/evaluate_csv_essential.py`](evaluate/evaluate_csv_essential.py) — AUC + Precision 怎么算的（≈ 5 min）。
6. [`presentation/diagnostics_features.py`](presentation/diagnostics_features.py) — 诊断指标定义（≈ 5 min）。

---

## 7. 常见疑问 FAQ

**Q1. 为什么只改 Q、V，不改 K、O 或 FFN？**
- 经验上 Q/V 对 attention 的输出最有杠杆；K 与 O 改动会显著破坏 DINOv3 已有的稳定 attention pattern。本论文是"小心翼翼地动一点点"的研究，所以选了最保守的 Q/V 组合。

**Q2. 为什么 batch_size = 8？**
- 不是为了显存，而是为了让 InfoNCE 的负样本来自 7 张**不同场景**的图，避免负样本退化为同图近邻（同图近邻已经被 Safe-Radius 屏蔽）。

**Q3. AUC@5 不涨、PCA 不变，那 Precision 为什么涨 7pp？**
- LoRA 是一次低秩旋转：在原 DINOv3 特征空间里轻微"歪一下"，使得 MNN + USAC 在 NAVI 这种单物体几何里更容易选出对的对应。它不会把"几乎对的"变成"完美对的"，但能把"几乎错的"挪进可接受阈值——这正是 AUC@20 ↑ 而 AUC@5 不动的原因。

**Q4. ViT-L/16 比 ViT-S/16 好吗？**
- 微调后 AUC@20 上 L/16 (1.123) 略好于 S/16 (1.040)；Precision 几乎打平（23.71% vs 25.06%）。本论文不主张 "更大更好"，而是说**recipe 在两档 backbone 上都稳定有效**。

**Q5. 与原仓库 `Superglue/` 是什么关系？**
- 仅作参考实现/对比阅读，**不是本工作的一部分**，本论文也未与 SuperGlue 做基准对比。

---

## 8. 一张表收尾

| 你想看 | 去哪 |
|---|---|
| 训练 loop | [`finetune/train_lora_hf.py`](finetune/train_lora_hf.py) |
| LoRA 注入实现 | [`finetune/lora_hf.py`](finetune/lora_hf.py) |
| 损失实现 | [`finetune/loss.py`](finetune/loss.py) |
| 数据正样本生成 | [`finetune/dataset.py`](finetune/dataset.py) + [`finetune/generate_train_pairs.py`](finetune/generate_train_pairs.py) |
| 推理 + MNN | [`finetune/extract_and_match_hf.py`](finetune/extract_and_match_hf.py) |
| AUC / Precision 计算 | [`evaluate/evaluate_csv_essential.py`](evaluate/evaluate_csv_essential.py) |
| 一键训 + 评 | [`scripts/train_oneclick.sh`](scripts/train_oneclick.sh)、[`scripts/train_oneclick_middle.sh`](scripts/train_oneclick_middle.sh) |
| 一键诊断出图 | [`scripts/run_diagnostics.sh`](scripts/run_diagnostics.sh)、[`analysis.sh`](analysis.sh) |
| 论文正式稿 | [`main_refine.tex`](main_refine.tex) |
| 仓库地图 | [`readme.txt`](readme.txt) |
