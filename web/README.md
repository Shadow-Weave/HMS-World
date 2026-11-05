# hms-world Web

hms-world 的浏览器工作区支持项目会话、Agent 对话、文件上传、产物与分镜预览、渲染进度和串行配置。前端通过 Agent Loop 与 JSONL 事件流调用串行流程，与 CLI 共用 Observer → Timeline → Replanner 闭环。

项目安装、串行示例和功能边界见 [hms-world README](../README.md)。以下命令从仓库根目录执行。

## 安装与配置

```sh
uv sync --locked --python 3.12
cp configs/serial_demo_local_h3.example.yaml configs/serial_demo_local_h3.yaml
cp configs/agent.example.yaml configs/agent.local.yaml
npm --prefix web ci
```

先配置 `serial_demo_local_h3.yaml` 中的本地聊天、图像和视频服务，以及 H3 工作流路径。`agent.local.yaml` 的 `serial.config` 指向该配置；设置面板可修改这一文件路径。对话默认复用串行 `chat_model`，面板中的 Agent LLM 为可选覆盖项。两份个人配置都已被 Git 忽略。

故事或剧本先转为可审阅的串行文本计划；请求生成时，再按 Timeline 逐镜生成、观察并调整下一镜。继续渲染从保存的 Timeline 恢复。修改已有计划会新建会话，原产物保留。旧会话可查看，但需要新建串行计划才能再次渲染。

Agent 配置也支持 `HMS_WORLD_*` 环境变量。

## 启动开发服务器

```sh
./hms-world web
```

默认地址为 `http://127.0.0.1:4173`。可通过 `HMS_WORLD_WEB_HOST` 和 `HMS_WORLD_WEB_PORT` 设置监听地址与端口。

## 构建后启动

```sh
npm --prefix web run build
./hms-world web start
```

## 验证

```sh
npm --prefix web test
npm --prefix web run build
(cd web && ./node_modules/.bin/tsc --noEmit)
```
