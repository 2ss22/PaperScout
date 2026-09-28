from pathlib import Path

from streamlit.testing.v1 import AppTest

from research_agent import config, llm


def set_env_file(tmp_path, monkeypatch, text):
    path = tmp_path / '.env'
    path.write_text(text)
    monkeypatch.setattr(config, 'ENV_FILE', path)
    return path


def test_local_configuration_wins_over_old_process_values(tmp_path, monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'deepseek')
    monkeypatch.setenv('DEEPSEEK_MODEL', 'deepseek-flash')
    monkeypatch.delenv('VISION_BASE_URL', raising=False)
    monkeypatch.delenv('VISION_API_KEY', raising=False)
    set_env_file(tmp_path, monkeypatch, 'LLM_PROVIDER=groq\nGROQ_MODEL=qwen/qwen3.8-27b\nGROQ_API_KEY=test-groq-key\nVISION_MODEL=qwen/qwen3.8-27b\n')
    result = config.load_configuration({'LLM_PROVIDER': 'deepseek'})
    assert result.text.provider == result.vision.provider == 'groq'
    assert result.text.model == result.vision.model == 'qwen/qwen3.8-27b'
    assert result.vision.base_url == config.GROQ_BASE_URL
    assert result.vision.api_key == result.text.api_key == 'test-groq-key'
    assert result.vision.key_source == 'GROQ_API_KEY'
    assert 'test-groq-key' not in repr(result)


def test_running_context_stays_fixed_next_context_refreshes(tmp_path, monkeypatch):
    path = set_env_file(tmp_path, monkeypatch, 'LLM_PROVIDER=deepseek\nDEEPSEEK_MODEL=old-model\n')
    with config.configuration_scope():
        assert config.settings.model == 'old-model'
        path.write_text('LLM_PROVIDER=groq\nGROQ_MODEL=new-model\n')
        assert config.settings.provider == 'deepseek'
        with config.configuration_scope():
            assert config.settings.provider == 'groq'
        assert config.settings.provider == 'deepseek'
    with config.configuration_scope():
        assert config.settings.model == 'new-model'


def test_cached_client_changes_with_endpoint_key_and_model(tmp_path, monkeypatch):
    made = []
    def create(**kwargs):
        obj = object()
        made.append((kwargs, obj))
        return obj
    monkeypatch.setattr(llm, 'OpenAI', create)
    monkeypatch.delenv('VISION_BASE_URL', raising=False)
    monkeypatch.delenv('VISION_API_KEY', raising=False)
    path = set_env_file(tmp_path, monkeypatch, 'LLM_PROVIDER=deepseek\nDEEPSEEK_API_KEY=test-old\n')
    llm._text_client.cache_clear()
    llm._vision_client.cache_clear()
    try:
        with config.configuration_scope():
            old = llm.get_client()
        path.write_text('LLM_PROVIDER=groq\nGROQ_MODEL=qwen/qwen3.8-27b\nGROQ_API_KEY=test-new\nVISION_MODEL=qwen/qwen3.8-27b\n')
        with config.configuration_scope():
            current = llm.get_client()
            assert current is llm.get_client() and current is not old
            llm.get_vision_client()
        assert made[-1][0]['base_url'] == config.GROQ_BASE_URL
        assert made[-1][0]['api_key'] == made[-2][0]['api_key'] == 'test-new'
    finally:
        llm._text_client.cache_clear()
        llm._vision_client.cache_clear()


def test_separate_vision_endpoint_never_inherits_groq_key():
    values = {'LLM_PROVIDER': 'groq', 'GROQ_API_KEY': 'private', 'VISION_MODEL': 'vision',
              'VISION_BASE_URL': 'https://different.example/v1'}
    vision = config.load_vision_settings(values)
    assert vision.api_key == '' and vision.key_source == '尚未配置'


def test_sidebar_refreshes_both_models_and_key_source(tmp_path, monkeypatch):
    monkeypatch.delenv('VISION_BASE_URL', raising=False)
    monkeypatch.delenv('VISION_API_KEY', raising=False)
    path = set_env_file(tmp_path, monkeypatch, 'LLM_PROVIDER=deepseek\nDEEPSEEK_MODEL=deepseek-flash\n')
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py').run()
    assert not app.exception
    assert any('deepseek · deepseek-flash' in c.value for c in app.caption)
    path.write_text('LLM_PROVIDER=groq\nGROQ_MODEL=qwen/qwen3.8-27b\nGROQ_API_KEY=test-only\nVISION_MODEL=qwen/qwen3.8-27b\n')
    app.run()
    assert not app.exception
    captions = '\n'.join(c.value for c in app.caption)
    assert '模型服务：groq · qwen/qwen3.8-27b' in captions
    assert '视觉模型：groq · qwen/qwen3.8-27b' in captions
    assert '密钥来源：GROQ_API_KEY' in captions
    assert 'test-only' not in captions and 'deepseek-flash' not in captions
