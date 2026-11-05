# 串行生成架构

hms-world 基于 ViMax 的角色图、分镜、机位转场和单镜制作能力，使用统一的串行闭环管理跨镜连续性。CLI、Web 和 TUI 共用同一套生产实现。

## 入口与职责

| 模块 | 职责 |
| --- | --- |
| `main_serial_demo.py` | 加载本地配置，启动或恢复串行任务 |
| `agent_runtime/hms_world_adapters.py` | 将 Web/TUI 的规划、修改、生成请求转给串行流程 |
| `pipelines/serial_demo_pipeline.py` | 实体登记、Timeline、逐镜制作、观察与下一镜调整 |
| `pipelines/script2video_pipeline.py`、`production/` | 角色图、分镜拆解、机位、单帧、单镜视频与合成 |
| `continuity/` | 状态模型、字段来源、参考材料与检查点 |
| `agents/` | 角色与实体提取、选图、提示词、Observer 和 Replanner |
| `tools/` | ComfyUI FLUX/Qwen 图像与 H3 视频适配器 |
| `evals/` | ViMax-Bench 和已有视频评分 |

## 每镜执行顺序

```text
剧本 → 实体登记与分镜 → Timeline
  → 取出当前镜计划与已有状态
  → 选择身份图、历史材料和机位参考
  → 生成首帧，按需要生成尾帧
  → 生成本镜视频
  → 采样 → Observer → Timeline 合并
  → Replanner 保留或替换下一镜
  → 继续下一镜 → 合成视频
```

机位只能引用已经生成的早期镜头。机位切换仍可能生成转场视频并提取新视角；父镜头完全覆盖新机位内容时，直接复用该画面作为首帧。因此这些原生制作能力属于串行运行依赖。

`FramePromptWriter` 负责最终绘图提示词。原生候选说明只作为写作输入，不在 writer 完成后追加到 Provider prompt。尾帧生成使用本镜实际首帧；参考材料的选择规则见[连续性状态与视觉材料](continuity.md)。

## 文本审阅与恢复

Web/TUI 的 `plan` 只保存文本计划、实体登记和 Timeline，不生成媒体。用户请求生成后调用 `resume_preplanned`，它可以补齐尚未完成的角色图并继续制作。

修改已有计划会创建新会话，只复制仍有效的文本。实体登记会读取审阅后的分镜与镜头描述，旧媒体和观察状态不带入新计划。旧普通会话可查看，重新制作时需要新建串行计划。

CLI 的 `run(..., resume=True)` 从原输出目录恢复。恢复会校验实体与镜头编号、机位依赖、观察记录和采样文件；已有镜头的真实媒体与检查点是续跑依据。

## 主要产物

相对于一个串行输出目录：

| 路径 | 内容 |
| --- | --- |
| `characters.json`、`character_portraits/` | 角色定义与身份图 |
| `storyboard.json`、`camera_tree.json` | 分镜和机位依赖 |
| `continuity/entity_registry.json` | 本次任务的实体登记 |
| `continuity/timeline.json` | 待执行计划与已观察状态 |
| `continuity/image_records.jsonl` | 图片记录及其来源 |
| `shots/<idx>/` | 镜头描述、首尾帧、视频、观察采样与 Replan 回执 |
| `final_video.mp4` | 按故事顺序合成的视频 |

交互会话写入 `.working_dir/<session_id>/serial/`；CLI 和评测使用各自指定的输出目录。

## 当前边界

Observer 可能误读画面；状态可追溯不等于画面一定正确。Replanner 只处理下一镜，尚无自动重拍、多镜连锁改写或任意早期剧情检索。正式评测直接使用这套生产接口，生成结果由独立的 ViCLIP 评分入口计算一致性指标。
