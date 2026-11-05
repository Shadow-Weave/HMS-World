"""Local configuration factory contract."""
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import yaml


def test_script_factory_accepts_local_data_and_existing_file_config(tmp_path, monkeypatch):
    module = importlib.import_module("pipelines.script2video_pipeline")
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/serial_demo_local_h3.example.yaml").read_text())
    config["working_dir"] = str(tmp_path / "output")
    model, image, video = object(), object(), object()
    init_model = MagicMock(return_value=model)
    backend = MagicMock(return_value=SimpleNamespace(image_generator=image, video_generator=video))
    monkeypatch.setattr(module, "init_chat_model", init_model)
    monkeypatch.setattr(module.RenderBackend, "from_config", backend)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    from_data = module.Script2VideoPipeline.init_from_config_data(config)
    from_file = module.Script2VideoPipeline.init_from_config(str(path))

    for pipeline in (from_data, from_file):
        assert pipeline.chat_model is model
        assert pipeline.image_generator is image
        assert pipeline.video_generator is video
        assert pipeline.working_dir == config["working_dir"]
    assert init_model.call_args_list[0] == init_model.call_args_list[1]
    assert init_model.call_args.kwargs["api_key"] == config["chat_model"]["init_args"]["api_key"]
    assert backend.call_args_list[0] == backend.call_args_list[1]
