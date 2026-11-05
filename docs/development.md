# 开发安装与验证

以下命令从仓库根目录执行。Python 环境由 `pyproject.toml` 和 `uv.lock` 管理，Web 和 TUI 各自保留 `package.json` 与 `package-lock.json`。两个前端独立安装。

## 锁定安装

开发使用 Python 3.12、uv、Node.js 和 npm。项目允许 Python >=3.12；前端锁定工具的 Node engines 为 `^18.0.0 || ^20.0.0 || >=22.0.0`，TUI 的 tsx 要求 >=18。

```sh
uv sync --locked --python 3.12
npm --prefix web ci
npm --prefix ui ci
```

`uv sync` 默认包含已声明的 dev 组，提供 pytest。首次安装不需要本地凭据；实际调用模型前配置串行 YAML；Web/TUI 的 `agent.local.yaml` 通过 `serial.config` 引用它。

```sh
cp configs/serial_demo_local_h3.example.yaml configs/serial_demo_local_h3.yaml
cp configs/agent.example.yaml configs/agent.local.yaml
./hms-world tui
./hms-world web
```

真实机位转场的视频拆分使用 SceneDetect 的 FFmpeg 命令；执行该路径时需要可用的 FFmpeg。Python 的 MoviePy 媒体处理依赖继续由现有锁文件安装。

## 完整离线验证

```sh
uv lock --check
.venv/bin/python -m pytest -q

npm --prefix web test
npm --prefix web run build
(cd web && ./node_modules/.bin/tsc --noEmit)

npm --prefix ui test
(cd ui && ./node_modules/.bin/tsc --noEmit)
```

Python 套件覆盖公开接口、Provider、Agent 会话、制作与缓存、检查点恢复、Observer 补看、稀疏状态合并、历史来源、下一镜 Replan，以及 ViMax-Bench 入口与一致性评分。Web 测试同时覆盖浏览器侧逻辑和 Node 服务；TUI 测试覆盖行号映射、斜杠命令和工作区元数据。

这些检查不调用真实模型生成或 ViCLIP 模型推理。需要实际生成或评测时，参阅 [hms-world README](../README.md) 和[评测说明](../evals/README.md)中的当前命令与配置。本地串行配置不使用旧网关配置或云端凭据。

## Provider 与媒体依赖

当前图像 Provider 为 ComfyUI FLUX.2 klein 和 Qwen，视频 Provider 为 ComfyUI H3。配置使用对应模块的完整 `class_path`；聊天服务使用 OpenAI 兼容接口。所有生成端点都须为本地回环地址，远程部署通过 SSH 转发接入。

SceneDetect 和 FFmpeg 仍用于机位转场，MoviePy 用于视频处理与合成。更新依赖时同步更新所属锁文件，并在不含个人配置和已有依赖目录的源码副本中验证安装与测试。
