# hms-world: World Memory for Video Generation

**Build memory from generated footage, and use it to inform what comes next.**

hms-world maintains an evolving world memory for multi-shot video generation: what a character is holding, where a prop was placed, and how a scene has changed. It stores known state, tracks where each field came from, and retains historical footage samples. Later shots read this memory to select visual references, write image prompts, and revise plans.

The project's focus is how memory is represented, written, updated, and read during generation. ViMax supplies character portraits, storyboards, camera transitions, and shot production. hms-world adds the memory that carries observed state into later shots, with cross-shot continuity as its application goal.

The repository's main documentation is the [hms-world README](README.md). Run commands from the repository root. Use `./hms-world` to launch the interactive interfaces; project environment variables use the `HMS_WORLD_*` prefix.

## Install

Run these commands from this directory with Python 3.12+ and uv installed:

```sh
uv sync --locked --python 3.12
```

Video processing requires FFmpeg. Web and TUI also require Node.js and npm; Node.js 22 is recommended.

## Run the serial continuity example

```sh
cp configs/serial_demo_local_h3.example.yaml configs/serial_demo_local_h3.yaml
```

Configure a local vision-capable chat endpoint, a ComfyUI image endpoint, and a ComfyUI H3 video endpoint. Replace the placeholder `video_generator.init_args.workflow_path` with your H3 API workflow JSON path. The supplied example uses FLUX.2 klein for images. Model services must be deployed separately; remote services can be forwarded to loopback addresses through SSH.

Keep `local_only: true` and use loopback endpoints.

```sh
.venv/bin/python main_serial_demo.py \
  --config configs/serial_demo_local_h3.yaml \
  --script-file examples/serial_demo_three_shot.txt \
  --user-requirement-file examples/serial_demo_three_shot_requirements.txt \
  --style "3D CGI animation" \
  --output .working_dir/serial_demo
```

To resume, use the original configuration, style, and output directory, add `--resume`, and omit the script and requirements arguments. The timeline checkpoint is stored at `.working_dir/serial_demo/continuity/timeline.json`; successful runs print the assembled video's path.

## Interactive workflows

Web/TUI use the same serial plan, observation and replanning loop as the CLI. They accept story ideas or screenplay text.

```sh
cp configs/agent.example.yaml configs/agent.local.yaml
```

Prepare the serial YAML above first. Set `serial.config` to that YAML, or use `HMS_WORLD_SERIAL_CONFIG`. Conversation uses its `chat_model` by default; optional `llm` settings override only the conversation model. Image, video and continuity roles all use the serial configuration.

Planning writes reviewable text and a Timeline without generating media. Rendering resumes that Timeline under `.working_dir/<session_id>/serial/`. Revisions create a new session and preserve the original media. Legacy sessions remain viewable; start a new serial session with the agreed story text to render them again.

For Web, install its dependencies and start the server:

```sh
npm --prefix web ci
./hms-world web
```

Open `http://127.0.0.1:4173`; see the [Web guide](web/README.md) for build and server options.

For TUI:

```sh
npm --prefix ui ci
./hms-world tui
```

## Evaluation and checks

```sh
.venv/bin/python evals/vimax/runner.py --list
.venv/bin/python evals/vimax/runner.py 29 --validate-only
.venv/bin/python -m pytest -q
```

Listing and validating benchmark data do not call model providers. See the [evaluation guide](evals/README.md) for generation, resume, and scoring requirements.

A historical ViMax-Bench development run, scored on October 20–22, 2026, covers all **35 stories / 437 shots**: **IC 0.6264, CC 0.6075, GC 0.6132**, averaged equally across stories. The project's current model configuration uses Qwen3-VL-30B-A3B, FLUX.2 klein 9B Base and ComfyUI H3. See the [results and configuration](README.md#benchmark-结果) and [per-story scores](evals/results/vimax-bench-20261022.json).

The ViMax baseline has completed **4/35 stories (42 shots)**, with IC **0.6798**, CC **0.6812** and GC **0.6807**. On those same four stories, hms-world scores **0.6888 / 0.6951 / 0.6927**. See the [partial baseline export](evals/results/vimax-baseline-partial-20261029.json) for case IDs, full scores and provenance. This subset comparison uses historical outputs from each run.

Observer output and generated frames can be incorrect. Replanning affects only the next shot; automatic retakes and multi-shot propagation are not implemented.

## Attribution and license

hms-world builds on HKUDS's ViMax video generation framework. The source retains its [MIT license](LICENSE).
