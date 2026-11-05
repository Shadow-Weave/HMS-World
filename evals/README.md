# hms-world 评测入口

ViMax-Bench 使用正式生产流水线，评测代码集中在此目录。应用代码不导入 evals。

先按 [hms-world README](../README.md) 安装依赖并配置本地模型服务。以下命令从仓库根目录执行。当前串行配置为 `configs/serial_demo_local_h3.yaml`。

## ViMax-Bench

```sh
.venv/bin/python evals/vimax/runner.py --list
.venv/bin/python evals/vimax/runner.py 29 --validate-only
.venv/bin/python evals/vimax/runner.py 29 --prepare-only
.venv/bin/python evals/vimax/runner.py 29 --execute
.venv/bin/python evals/vimax/runner.py 29 --resume \
  --output-dir /absolute/path/to/existing/run
```

`--list`、`--validate-only` 不调用 Provider。`--prepare-only` 只准备已有分镜的文字计划；`--execute` 顺序生成；`--resume` 复用原 manifest 和生产检查点。35 个故事与索引保存在 [`vimax/data`](vimax/data)，原始 first_frame 和 video_prompt 字符串继续用于制作。

适配后的 `visual_desc` 同时包含官方首帧描述与完整视频动作，供机位规划和 Replanner 理解整镜；`ff_desc` 和 `motion_desc` 分别保留官方原文。此数据集没有独立尾帧目标，因此使用单首帧 `small` 路径，`lf_desc` 为空，动作的结束要求由 `motion_desc` 表达。Replanner 按实际启用的帧条件检查一致性，不把空的非执行尾帧字段判作失败。旧运行的分镜、状态和评分不会自动回写为新适配结果。

### 对已有视频评分

`vimax/score.py` 使用 ViCLIP-L / InternVid-10M-FLT 给一个完整案例的已有镜头评分，不调用生成 Provider。评分依赖放在独立的 `.working_dir/viclip-runtime/.venv`；这套环境不随仓库提供，需要单独准备。已有环境中，该目录的 `source_manifest.json` 记录官方源码版本和权重 SHA256，`requirements.freeze.txt` 记录实际安装版本。模型及源码均保存在该目录，不进入 Git。

已准备上述评分环境后可这样使用：

```sh
.working_dir/viclip-runtime/.venv/bin/python evals/vimax/score.py \
  /absolute/path/to/existing/run/29-scientists_lab_collaboration \
  --device mps \
  --output-dir .working_dir/vimax_scores/case29
```

也可传 `--device cpu`，或用默认的 `auto` 自动选择可用的 MPS/CPU。评分沿用官方 ViCLIP 的 8 帧、224×224、RGB 归一化；为兼容 MPS，先在 CPU 转成相同的 float32 输入再传到设备。每镜先编码一次，再计算不同镜头的余弦相似度：IC 为官方 `scene_num` 组内平均，CC 为组间平均，GC 为所有不同镜头对的平均，均不计自身配对。生成阶段的物理场景绑定不替换评分分组。

输出包括 `first_shot_timing.json`、`scores.json`、特征数组和相似度矩阵。时间分别记录模型加载、视频解码、预处理和同步后的推理；首镜计时包含其首次 GPU 执行开销。评分命令每次输出一个故事的分数。

### 已有评分结果

2026-10-20 至 2026-10-22 评分的历史开发批次覆盖 35 个故事、437 个镜头。逐故事分数与分组均值保存在 [`results/vimax-bench-20261022.json`](results/vimax-bench-20261022.json)，总分按故事等权平均：IC **0.6264**、CC **0.6075**、GC **0.6132**。

分数对应上述历史批次。项目最终模型配置为 Qwen3-VL-30B-A3B、FLUX.2 klein 9B Base 和 ComfyUI H3。指标定义、完整表格和配置参数见[项目 README](../README.md#benchmark-结果)。

ViMax baseline 已完成案例 1、13、17、19，共 4/35 个故事、42 个镜头。部分结果为 IC **0.6798**、CC **0.6812**、GC **0.6807**；hms-world 在相同 4 个故事上的均值为 **0.6888 / 0.6951 / 0.6927**。逐故事分数、原生流程版本及对照范围见 [`results/vimax-baseline-partial-20261029.json`](results/vimax-baseline-partial-20261029.json)。部分 baseline 与 35 故事总分使用不同案例集合，应使用 README 中的同案例表进行比较。

### 批量运行

同一个 runner 接受多个故事 ID，或用 `--all` 选择全部故事。案例按顺序运行，每个案例内继续遵循串行闭环：

```sh
.venv/bin/python evals/vimax/runner.py 1 2 3 --execute
```

项目配置和默认输出以仓库根目录为基准。内置数据目录为 `evals/vimax/data`；自定义数据可通过 `--benchmark-root` 指定。续跑使用同一输出目录与原 manifest，已生成的媒体和检查点保留。
