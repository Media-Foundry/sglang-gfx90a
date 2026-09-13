"""V4.1 encoder dispatch without importing serving benchmark dependencies."""

import copy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from sglang.srt.entrypoints.openai import chat_encoding, encoding_dsv41
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat


class RecordingTokenizer:
    chat_template = None
    bos_token_id = 0

    def encode(self, text):
        self.last_text = text
        return [ord(c) for c in text]


def encode_request(request):
    serving = OpenAIServingChat.__new__(OpenAIServingChat)
    serving.chat_encoding_spec = "dsv41"
    serving._dsv41_effort_profile = ({"low": 50, "high": 75, "max": 100}, "high")
    tokenizer = RecordingTokenizer()
    serving.tokenizer_manager = SimpleNamespace(tokenizer=tokenizer)
    mode = "thinking" if request.chat_template_kwargs.get("thinking", False) else "chat"
    messages = [m.model_dump() for m in request.messages]
    tools = [t.model_dump() for t in request.tools] if request.tools else None
    serving._encode_messages(copy.deepcopy(messages), request, mode, tools)
    return tokenizer.last_text


def test_architecture_dispatch_does_not_confuse_v4_and_v41():
    tokenizer = RecordingTokenizer()
    for arch, expected in (("DeepseekV41ForCausalLM", "dsv41"), ("DeepseekV4ForCausalLM", "dsv4")):
        assert chat_encoding.resolve_chat_encoding_spec(
            hf_config=SimpleNamespace(architectures=[arch]), tokenizer=tokenizer
        ) == expected
    assert chat_encoding.resolve_chat_encoding_spec(
        hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
        tokenizer=tokenizer, tool_call_parser="deepseekv41"
    ) == "dsv41"


@pytest.mark.parametrize("effort,budget", [(1, 1), (75, 75), (100, 100), ("low", 50), ("high", 75), ("max", 100)])
def test_numeric_effort_and_aliases(effort, budget):
    request = ChatCompletionRequest(model="test", messages=[{"role": "user", "content": "Hi"}], reasoning_effort=effort)
    text = encode_request(request)
    assert f"<｜System｜>Reasoning Effort: {budget} (range 1-100," in text
    assert text.endswith("<｜Assistant｜><think>")


def test_nested_budget_and_plain_chat():
    request = ChatCompletionRequest(model="test", messages=[{"role": "user", "content": "Hi"}], reasoning={"effort": 75})
    assert type(request.reasoning_effort) is int
    assert "Reasoning Effort: 75" in encode_request(request)
    plain = ChatCompletionRequest(model="test", messages=[{"role": "user", "content": "Hi"}], chat_template_kwargs={"thinking": False})
    assert encode_request(plain) == "<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜></think>"


@pytest.mark.parametrize("effort", [True, 0.995, 75.5, 101, -1])
def test_invalid_numeric_effort(effort):
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="test", messages=[{"role": "user", "content": "Hi"}], reasoning_effort=effort)


def test_tools_and_mid_conversation_system():
    request = ChatCompletionRequest(
        model="test", chat_template_kwargs={"thinking": False},
        messages=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"},
                  {"role": "system", "content": "Now answer briefly."}],
        tools=[{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}],
    )
    text = encode_request(request)
    assert "<｜DSML｜ calls>" in text and "<｜DSML｜tool_calls>" not in text
    assert text.endswith("<｜System｜>Now answer briefly.<｜Assistant｜></think>")


def test_offline_helper_uses_same_v41_encoder():
    tokenizer = RecordingTokenizer()
    messages = [{"role": "user", "content": "Hi"}]
    ids = chat_encoding.encode_simple_chat(tokenizer=tokenizer, spec="dsv41", messages=messages, thinking_mode="thinking")
    expected = encoding_dsv41.encode_messages(messages, thinking_mode="thinking")
    assert ids == [ord(c) for c in expected]


def test_local_checkpoint_alias_profile_is_read_without_execution(tmp_path):
    enc = tmp_path / "encoding"
    enc.mkdir()
    (enc / "encoding.py").write_text(
        "raise RuntimeError('must not execute checkpoint source')\n"
        "REASONING_EFFORT_MAPPINGS: dict = {'low':50, 'high':75, 'max':100}\n"
        "DEFAULT_REASONING_EFFORT = 'high'\n"
    )
    profile = chat_encoding.resolve_dsv41_effort_profile(str(tmp_path))
    assert chat_encoding.dsv41_effort_budget(None, profile) == 75
    assert chat_encoding.dsv41_effort_budget("low", profile) == 50
    assert chat_encoding.dsv41_effort_budget(42, profile) == 42


def test_v41_tool_detector_is_registered():
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from sglang.srt.function_call.deepseekv41_detector import DeepSeekV41Detector

    parser = FunctionCallParser([], "deepseekv41")
    assert isinstance(parser.detector, DeepSeekV41Detector)


def test_v41_registered_parser_reads_spaced_dsml():
    import json
    from sglang.srt.entrypoints.openai.protocol import Tool
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    tool = Tool(type="function", function={"name": "lookup", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}})
    parser = FunctionCallParser([tool], "deepseekv41")
    _, calls = parser.parse_non_stream(
        '<｜DSML｜ calls>\n<｜DSML｜ invoke name="lookup">\n'
        '<｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>\n'
        '</｜DSML｜ invoke>\n</｜DSML｜ calls>'
    )
    assert len(calls) == 1 and calls[0].name == "lookup"
    assert json.loads(calls[0].parameters) == {"city": "Paris"}


@pytest.mark.parametrize("body", [
    {"object": "error", "message": "Invalid effort", "code": 400},
    {"error": {"message": "Invalid effort", "code": 400}},
    {"detail": [{"msg": "Invalid effort"}]},
])
def test_api_probe_accepts_explicit_error_schemas(tmp_path, body):
    import io
    import json
    from scripts.rocm.check_dsv41_chat_api import Probe

    response = io.BytesIO(json.dumps(body).encode())
    response.status = 400
    probe = Probe("http://unused", tmp_path, 1)
    probe.opener = SimpleNamespace(open=lambda *a, **k: response)
    assert probe.call("error", {}, expected_status=400) is None
    assert probe.results[0]["protocol_passed"]


@pytest.mark.parametrize("stream,body", [
    (False, '{"choices": [], "usage": {"completion_tokens": 0}}'),
    (False, '{"choices": [{"message": {"content": ""}, "finish_reason": "stop"}], "usage": {"completion_tokens": 1}}'),
    (True, 'data: {"choices": [{"index": 0, "delta": {"content": "Paris"}, "finish_reason": "stop"}]}\n\n'),
])
def test_api_probe_rejects_empty_or_truncated_success(tmp_path, stream, body):
    import io
    from scripts.rocm.check_dsv41_chat_api import Probe

    response = io.BytesIO(body.encode())
    response.status = 200
    probe = Probe("http://unused", tmp_path, 1)
    probe.opener = SimpleNamespace(open=lambda *a, **k: response)
    with pytest.raises(AssertionError):
        probe.call("invalid-success", {"stream": stream})
    assert not probe.results[0]["protocol_passed"]
