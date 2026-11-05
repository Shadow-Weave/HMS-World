import pytest
import yaml

from agent_runtime.config import load_agent_config, llm_model, llm_api_key, llm_base_url, serial_config_path


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch):
    load_agent_config.cache_clear()
    for key in ('HMS_WORLD_LLM_MODEL', 'HMS_WORLD_LLM_API_KEY', 'HMS_WORLD_API_KEY', 'HMS_WORLD_LLM_BASE_URL', 'HMS_WORLD_SERIAL_CONFIG'):
        monkeypatch.delenv(key, raising=False)
    yield
    load_agent_config.cache_clear()


def test_conversation_reuses_the_serial_chat_config(tmp_path):
    (tmp_path / 'configs').mkdir()
    path = tmp_path / 'configs/serial_demo_local_h3.yaml'
    path.write_text(yaml.safe_dump({'chat_model': {'init_args': {'model': 'local-vision', 'base_url': 'http://localhost:9000/v1', 'api_key': 'local'}}}))
    assert llm_model(tmp_path) == 'local-vision'
    assert llm_base_url(tmp_path) == 'http://localhost:9000/v1'
    assert llm_api_key(tmp_path) == 'local'
    assert serial_config_path(tmp_path) == path


def test_agent_override_and_environment_precedence(tmp_path, monkeypatch):
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/agent.local.yaml').write_text(yaml.safe_dump({'llm': {'model': 'chat-only', 'api_key': 'test-key'}, 'serial': {'config': 'configs/custom.yaml'}}))
    assert llm_model(tmp_path) == 'chat-only'
    assert llm_api_key(tmp_path) == 'test-key'
    assert serial_config_path(tmp_path) == tmp_path / 'configs/custom.yaml'
    monkeypatch.setenv('HMS_WORLD_LLM_MODEL', 'env-chat')
    monkeypatch.setenv('HMS_WORLD_SERIAL_CONFIG', str(tmp_path / 'external.yaml'))
    assert llm_model(tmp_path) == 'env-chat'
    assert serial_config_path(tmp_path) == tmp_path / 'external.yaml'
