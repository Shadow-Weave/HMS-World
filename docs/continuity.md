# 连续性状态与视觉材料

本文描述 Observer、Timeline 和历史材料的当前合同；完整流程见[串行生成架构](architecture.md)。

## 已实现的信息流

一个镜头的 3×4 board 包含实际选中的实体参考和按时序采样的画面。Observer 中性地阅读实际过程并记录末次采样时刻的状态，可以自行请求最多两轮原尺寸采样帧补看，最多三次模型调用，最终只返回一份本镜参与实体的稀疏 `WorldState` 更新。它不负责重新输出整个世界，也不比较计划与实际结果。`Timeline.mark_observed` 延续上一份快照、补入首次参与实体的已知初始内容，再应用本镜更新，形成该位置唯一的完整已知状态。

Observer 接收 `PARTICIPANT_BASELINE`、`PARTICIPANT_SCOPE`、`REGISTRY_ENTITY_NAMES` 和一张 board。baseline 与名称只包含当前 `ShotPlan` 的参与实体；图片直接标注 `CHAR`、`SCENE` 与带时间的 `T0`、`T1` 等时序帧编号，不再发送 reference/temporal slot map 或素材 image ID。board 已清楚建立的事实直接记录；只有模型不确定或对比理解冲突，且原图有望解答时才补看，不按字段类别触发。`inspect_samples(frames=[...])` 直接使用图上的 T 编号，例如 `frames=[9]` 查看 T9；每次可指定一至三帧，以原尺寸附图，不重新解码视频或增加采样。当前接口不发送动作计划、`PREVIOUS_WORLD_STATE` 或 `EXECUTED_VISUAL_PLAN`。

稀疏输出与完整快照共用 `WorldState` 类型，但语义不同：

| Observer 输出 | Timeline 应用方式 |
| --- | --- |
| 省略实体、实体值为 `null`、空实体对象 | 不更新该实体 |
| 省略内容字段 | 不更新该字段；证据不足时使用此方式 |
| 显式字段值为 `null` | 清除旧字段值及其来源；仅用于确认该属性不存在 |
| `holding: []` | 确认未持有本镜登记道具，覆盖旧持物列表；其他接触由 pose 描述 |
| 非空字段值 | 更新该字段；值未变时保留原来源 |
| camera 省略或为 `null` | 本镜 camera 未知，不沿用上一镜 camera |

完整快照中的 `null` 表示当前没有已知值；它不能反过来当作稀疏更新中的“未知占位符”。已出现但不参与本镜的实体由 Timeline 保留，Observer 不输出它们。人物持物 ID 必须属于本镜投影的道具集合。

```text
ObservedPosition.state
  + value-free field sources
  → StateFieldLineage(entity / field / value / source shot)
  → source shot 的 observation_samples.json
  → 该 shot 全部正式 shot_sample ImageRecords
  → 每个来源镜头的尾 sample 和一张可复现的随机 sample
  → selector 的可选 history candidates
```

`ObservedShotSource(shot_idx)` 是某非空字段首次新增或最近变化的 observed shot lineage；它不带 `sample_index`，也不是具体图片证据。更新值与合并前的已知值相同，来源保持不变；新增或变化的非空字段才指向当前 shot，显式清空字段会移除其来源。

`observation_samples.json` 是单镜的有序输入采样清单：`index`、`time_seconds`、`image_id`。它没有 state、summary 或 provenance。每张 sample 对应一个 `shot_sample` ImageRecord，记录真实图、shot、时间和固定客观说明；记录不保存或概括 WorldState。Observer 不再生成独立的评判报告。

因此字段 lineage 只能定位到 source shot。程序用本镜已绑定的道具实体 ID 过滤字段来源，再读取保留镜头的正式 samples；上一镜仍被人物拿着的道具可进入状态投影，但不会仅因被携带而成为本镜历史图候选。每个保留镜头只向 selector 提供最后一张正式采样帧和一张可复现的随机采样帧。任何一帧都不被预先宣称为字段的完整证明。

合并后的 `state` 进入 `ObservedPosition`：人物的 outfit、expression/emotion、pose、叙事 position、orientation、held registered props；道具 position/condition；场景的可见布置与状态；以及本镜 CameraState scale/angle/composition。人物、道具和场景的非空字段进入 field lineage；camera 不进入字段来源索引或历史候选定位。没有独立的 Observer notes、差异报告或质量报告。

## 当前 selector

现有 `ContinuityImageRetriever` 的一次模型调用同时整理：

- 每个可见人物恰好一张 portrait；
- history sample IDs；
- ViMax 原生候选 IDs。

history 与 native 共享 portrait/fixed 之外的剩余 `top_k`，每个 source shot 最多选一张 history sample，也可完全不选。模型会看到当前 endpoint、相关字段 lineage、已预留 fixed material 的职责，以及筛出镜头的尾帧和随机帧；fixed material 不供它重复选择。

被选中的 history 图只能用于画面中实际可见、且与 lineage 相关的人物/场景/道具连续性。它不负责当前镜的姿势、构图、景别或表演，也不是全部字段的 proof。该流程没有新增 resolver、摘要 Agent、关键帧 Agent 或额外 LLM 调用。

ViMax native candidate 仍只来自 ViMax 实际交给 selector 的候选；continuity 不从 Camera 关系推断或补造它。制作尾帧时，本镜首帧仍是 required material。

## Observer board 的历史参考

board 只读取已存在的首/尾 endpoint 选图回执，不重跑 selector：

- 实际选中的 portrait 可作为人物身份参考；
- 实际选中的 history sample 仅在其 group lineage 包含当前 scene entity 时，最多一张可作为场景参考；
- native candidate 和 current-first-frame/fixed material 不进入 board；
- 没有实际选中就不补、不推断。

当前 `build_frame_material_context` 只用本镜绑定的道具字段构造 history lineage，不含 scene lineage，因此这条自动路径不会形成满足 scene-reference 条件的候选。不能把有条件的 scene 分支写成每镜实际提供场景参考。

board 本身不写入 Timeline、历史材料或图片记录。参考格占用前导槽位，其余为时序格；总共十二格且至少留两格时序采样。

## 首帧提示词与当前限制

`FramePromptWriter` 接收一个 endpoint 的静态目标、已知字段 lineage 和已选图片，写一条静态绘图提示词。明确目标指令优先，已有状态补足目标未替换的连续性。portrait 只锚定身份，history 只提供可见且相关的连续性，native 是普通可选参考；尾帧另有本镜实际首帧作为必需起点材料。

串行流程将 writer 返回的最终提示词原样交给 Provider，不再追加原生候选的说明。候选材料包含的旧状态、姿势或构图仍可能影响生成结果，需要结合真实画面判断连续性与质量。

## Replanner 与恢复

Replanner 同时接收上一镜剧本、Timeline 合并后的完整已知终态，以及从该镜 manifest 读取的最后实际 sample。Replanner 自己比较计划与实际结果并判断后果，输出仍是 `NoChange | Replacement`；它可在 reason 提到补拍关切，但没有自动补拍路径。该尾图只标为 “Previous shot's actual terminal sampled frame”；系统允许电影剪辑，不要求逐帧无缝，也不把这张图说成终态或字段证明。

resume 验证每个 observed shot 的 manifest 条目均有同 shot/time、路径有效的正式 `shot_sample` record。不把 sample 数量或内容与 WorldState 对齐，也不读取额外的 Observer 报告。

## 未实现

本期没有从自然语言任意定位更早剧情位置的能力，也没有规模化长历史候选的索引或加速。不能用摘要、尾帧假设、额外状态副本或另一个 Agent 替代这些未实现能力。
