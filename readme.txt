================================================================
  CV-Project · DINOv3 + LoRA + HardInfoNCE + Safe Radius
  NAVI 稠密匹配 / 位姿估计 微调实验仓库
================================================================

【一句话总览】
  本仓库基于 HuggingFace 版 DINOv3 (ViT-S/16 与 ViT-L/16)，在 NAVI
  数据集上以 LoRA(r=4, α=8, 仅 Q/V) + HardInfoNCE + 5-patch
  Safe-Radius 进行参数高效微调，并对 zero-shot vs LoRA 的特征做
  代表性 / 几何 / 层级 三类诊断分析。

  最终结果（NAVI test, 908 pairs, 1024² 评估）：
	ViT-S/16  Precision 17.38% → 25.06% (+7.7pp), AUC@20 0.436 → 1.040
	ViT-L/16  Precision 16.75% → 23.71% (+7.0pp), AUC@20 0.456 → 1.123
  AUC@5 在两个 backbone 上均未提升（已在论文 §4.5 Limitations
  中诚实给出）。

================================================================
					   一、目录结构地图
================================================================

cv-project/
├── readme.txt                # 本文件，仓库使用入口
├── PAPER_GUIDE.md            # 论文辅助阅读手册（推荐第三者先读这份）
├── main.tex                  # 旧版完整稿（含 phase1 失败分析等历史内容）
├── main_refine.tex           # ★当前正式版论文（仅 phase2 + 诊断 + 局限）
├── references.bib            # bib 引用
├── requirements.txt          # Python 依赖（pip 版）
├── analysis.sh               # 诊断脚本的常用调用速查
├── learn.md                  # 个人学习笔记 / 思路过程，可选阅读
│
├── dinov3_weights/           # ★ HuggingFace DINOv3 预训练权重根目录
│   ├── dinov3-small/         #   ViT-S/16 (会被脚本自动定位)
│   └── dinov3-middle/        #   ViT-L/16
│
├── datasets/                 # 测试数据 + pair 列表
│   ├── tiny/                 #   极小 smoke-test 用集合
│   ├── test/navi_resized/    #   NAVI test 图（长边已 resize 到 1024）
│   └── *_pairs.txt           #   评估用 pair 文件，格式见后
│
├── finetune/                 # ★ 训练核心代码（HF 路径优先）
│   ├── lora_hf.py            #   LoRA 注入：把 r/α 包到 HF 的 Q/V 投影
│   ├── model_hf.py           #   HF Dinov3ViTModel 的封装 + patch token 抽取
│   ├── dataset.py            #   NAVI / ScanNet pair 数据集 + 深度回投正样本
│   ├── loss.py               #   HardInfoNCE + Safe-Radius mask
│   ├── train_lora_hf.py      #   ★ 主训练入口（被一键脚本调用）
│   ├── extract_and_match_hf.py  # 推理：抽 patch 特征 → MNN 匹配 → CSV
│   ├── extract_lora.py       #   把 LoRA 权重单独导出 / 合并
│   ├── generate_train_pairs.py  # 由 NAVI 原始数据生成训练 pair 列表
│   ├── navi_train_pairs*.txt #   预生成的训练 pair 列表
│   ├── config.py             #   超参 / 路径默认配置
│   ├── smoke_test_hf.py      #   1 step 烟雾测试，验证管线打通
│   └── (lora.py / model.py / extract_and_match.py / train.py /
│        train_lora.py 是早期非-HF 版本，已不再用作正式实验入口)
│
├── evaluate/
│   └── evaluate_csv_essential.py  # ★ 把 matching CSV → AUC@{5,10,20} + Precision
│
├── scripts/                  # 一键脚本集合
│   ├── setup_env.sh / setup_env_conda.sh   # 安装环境
│   ├── build_navi_pairs.sh                 # 调用 generate_train_pairs.py
│   ├── make_tiny_dataset.py                # 抽极小子集做 smoke test
│   │
│   ├── lib_oneclick_pipeline.sh            # ★ 训练→分epoch评估→汇总 的实现库
│   ├── train_oneclick.sh                   # ViT-S/16 一键训练+评估
│   ├── train_oneclick_middle.sh            # ViT-L/16 一键训练+评估
│   │
│   ├── lib_phase1_pipeline.sh              # phase1 早期失败实验的实现库（保留以备查）
│   ├── train_phase1.sh / train_phase1_middle.sh   # phase1 入口（已不在 main_refine.tex 中使用）
│   │
│   ├── build_eval_summary.py               # 汇总每个 epoch 的指标 → summary.tsv
│   └── run_diagnostics.sh                  # ★ 一键跑全部诊断（PCA / features / layer4 / per_epoch）
│
├── presentation/             # ★ 诊断 / 可视化 / 汇报材料
│   ├── plot_per_epoch.py            # 画 per-epoch 三指标曲线（small vs middle）
│   ├── plot_loss_convergence.py     # 训练 loss 收敛曲线
│   ├── plot_results.py              # 最终对比柱状图
│   ├── pca_visualizer.py            # 把 patch token 投到 RGB 三通道做可视化
│   ├── diagnostics_features.py      # ★ 表征级诊断：intra-cos / pos-cos / eff-rank / neigh-dom
│   ├── diagnostics_layer4.py        # 中间层(Layer-4) 与 pose 角度的关系探测
│   ├── ppt_materials.md             # 汇报用素材清单
│   ├── ppt_detailed_script.md       # 汇报详细讲稿
│   └── result/                      # ★ 上述脚本输出的图与 tsv（论文直接引用）
│       ├── per_epoch_small_vs_middle.png
│       ├── diag_small/    (bars / hist_intra_cos / hist_pos_cos / pca_*_compare …)
│       └── diag_middle/   (上述 + layer4_scatter / layer4_pose_hist)
│
├── output/                   # ★ 训练 / 评估产物
│   ├── navi_small/
│   │   ├── lora_ckpt/checkpoint_latest.pth   # 最新 LoRA 权重
│   │   └── eval_per_epoch/summary.tsv        # 每个保存 epoch 的 AUC/Precision
│   └── navi_middle/   (同上)
│
└── Superglue/                # 参考实现（非本工作主体，仅留作对比阅读）

================================================================
			  二、关键模块（按重要性排序）
================================================================

【★★★ 训练核心三件套】
  finetune/lora_hf.py       —— LoRA 实现：r=4, α=8, 仅作用于 Q,V
  finetune/loss.py          —— HardInfoNCE + Safe-Radius (mask 半径=5)
  finetune/train_lora_hf.py —— 主训练循环 + 每 N epoch 评估保存

【★★★ 推理与评估】
  finetune/extract_and_match_hf.py —— DINOv3 + LoRA 抽特征 → 互最近邻匹配 → CSV
  evaluate/evaluate_csv_essential.py —— 由 CSV 算 AUC@{5,10,20}+Precision

【★★ 一键管线】
  scripts/train_oneclick.sh         —— ViT-S/16 整套（训→每3 epoch评→汇总）
  scripts/train_oneclick_middle.sh  —— ViT-L/16 整套
  scripts/lib_oneclick_pipeline.sh  —— 上面两者共用的实现库

【★★ 诊断与可视化】
  scripts/run_diagnostics.sh        —— 一把跑完 4 类诊断（见 analysis.sh 速查）
  presentation/diagnostics_features.py —— 表征级 6 项指标
  presentation/diagnostics_layer4.py   —— 中间层 pose-disentanglement 探测
  presentation/pca_visualizer.py       —— PCA→RGB 可视化

【★ 数据准备】
  finetune/generate_train_pairs.py + scripts/build_navi_pairs.sh
  scripts/make_tiny_dataset.py —— 极小数据集，用于 smoke test

================================================================
				  三、典型工作流
================================================================

(0) 准备环境
	bash scripts/setup_env.sh           # 或 setup_env_conda.sh

(1) 烟雾测试（验证管线）
	python finetune/smoke_test_hf.py

(2) 一键训练 + 每 3 epoch 评估 + 汇总（任选一档 backbone）
	bash scripts/train_oneclick.sh           # ViT-S/16
	bash scripts/train_oneclick_middle.sh    # ViT-L/16
	产物：
	  output/navi_{small,middle}/lora_ckpt/checkpoint_latest.pth
	  output/navi_{small,middle}/eval_per_epoch/summary.tsv

(3) 诊断分析（生成论文 §4.4 所有图）
	bash scripts/run_diagnostics.sh
	# 或单独跑某一项，见 analysis.sh

(4) 渲染论文
	latexmk -xelatex main_refine.tex      # 当前正式稿
	# main.tex 是含 phase1 历史内容的旧稿，仅供查阅

================================================================
			  四、pair 文件格式（评估用）
================================================================

每行一对：
  path_A path_B  exif_rotA exif_rotB
  KA_00..KA_08   KB_00..KB_08
  T_AB_00..T_AB_15
即：路径 + EXIF 旋转 + 两份 3x3 内参 + 4x4 相对外参（行优先）。
NAVI 长边已 resize 到 1024，ScanNet resize 到 640x480。

================================================================
				  五、产出与论文对应
================================================================

main_refine.tex 中的图 / 表 / 节 → 仓库文件
  Fig. per-epoch (§4.1)         ← presentation/result/per_epoch_small_vs_middle.png
  Table 最终指标 (§4.2)         ← output/navi_*/eval_per_epoch/summary.tsv 汇总
  Fig. hist_intra_cos (§4.4)    ← presentation/result/diag_*/hist_intra_cos_*.png
  Fig. hist_pos_cos  (§4.4)     ← presentation/result/diag_*/hist_pos_cos_*.png
  Fig. bars_summary  (§4.4)     ← presentation/result/diag_*/bars_summary_*.png
  Fig. PCA compare   (§4.4)     ← presentation/result/diag_*/pca_*_compare.png
  Fig. Layer4 (§4.4, 仅 L/16)   ← presentation/result/diag_middle/layer4_*.png
  §4.5 Limitations + §5 末段    ← 与上述诊断结果一一对应

================================================================
					 六、历史兼容说明
================================================================

  · finetune/ 中带 _hf 后缀的是当前正式 HuggingFace 路径；
	不带 _hf 的（lora.py / model.py / train.py / train_lora.py /
	extract_and_match.py）是早期非-HF 版本，已不在正式实验中使用，
	保留是为了和 main.tex 历史叙述对得上。
  · scripts/lib_phase1_pipeline.sh 与 train_phase1*.sh 对应论文
	旧稿 main.tex 中的 phase1 失败分析；当前正式稿 main_refine.tex
	已不依赖它们。

================================================================