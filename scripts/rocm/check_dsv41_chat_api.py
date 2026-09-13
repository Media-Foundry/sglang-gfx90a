#!/usr/bin/env python3
"""Proxy-free V4.1 chat/tool API smoke; never executes a model-requested tool.

Records raw JSON/SSE, requires completed nonempty responses, and checks a
synthetic tool-result round trip. This is an interface/semantic smoke, not a
general model-quality or performance benchmark. It does not test /v1/responses.
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


class Probe:
    def __init__(self, url, output_dir, timeout):
        self.url = url.rstrip("/")
        self.output_dir = output_dir
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.results = []

    def call(self, name, payload, *, expected_status=200):
        record = {"name": name, "request": payload, "protocol_passed": False}
        start = time.monotonic()
        try:
            request = urllib.request.Request(
                self.url + "/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                response = self.opener.open(request, timeout=self.timeout)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                record["status"] = response.status
                if response.status != 200 or not payload.get("stream"):
                    raw = response.read().decode()
                    record["raw_body"] = raw
                    body = json.loads(raw)
                    record["response"] = body
                    assert response.status == expected_status, raw
                    if expected_status != 200:
                        assert (
                            body.get("error") or body.get("detail")
                            or (body.get("object") == "error" and body.get("message"))
                        ), body
                        record["protocol_passed"] = True
                        return None
                    assert len(body.get("choices", [])) == 1, body
                    choice = body["choices"][0]
                    message = choice["message"]
                    finish = choice.get("finish_reason")
                    usage = body.get("usage", {})
                else:
                    events, text, reasoning, calls = [], "", "", {}
                    finish, done, usage = None, False, {}
                    record["events"] = events
                    for raw_line in response:
                        line = raw_line.decode().strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            done = True
                            break
                        event = json.loads(data)
                        events.append(event)
                        assert not event.get("error"), event
                        usage = event.get("usage") or usage
                        for choice in event.get("choices", []):
                            assert choice["index"] == 0, choice
                            delta = choice.get("delta", {})
                            text += delta.get("content") or ""
                            reasoning += delta.get("reasoning_content") or ""
                            finish = choice.get("finish_reason") or finish
                            for call in delta.get("tool_calls") or []:
                                item = calls.setdefault(call["index"], {
                                    "id": "", "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                })
                                if call.get("id"):
                                    assert item["id"] in ("", call["id"]), call
                                    item["id"] = call["id"]
                                function = call.get("function", {})
                                item["function"]["name"] += function.get("name") or ""
                                item["function"]["arguments"] += function.get("arguments") or ""
                    assert done, "SSE ended without [DONE]"
                    message = {"role": "assistant", "content": text,
                               "reasoning_content": reasoning,
                               "tool_calls": [calls[i] for i in sorted(calls)] or None}
                record.update(message=message, finish_reason=finish, usage=usage)
                assert finish in ("stop", "tool_calls"), f"Incomplete response: {finish}"
                assert isinstance(usage.get("completion_tokens"), int) and usage["completion_tokens"] > 0, usage
                assert message.get("content") or message.get("tool_calls"), message
                assert "｜DSML｜" not in (message.get("content") or ""), message
                record["protocol_passed"] = True
                return message
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["elapsed_s"] = time.monotonic() - start
            self.results.append(record)
            self.save(name, record)

    def save(self, name, value):
        with (self.output_dir / f"{name}.json").open("x") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30101")
    parser.add_argument("--model", default="/media/PM983/deepseek-v4.1-flash")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    probe = Probe(args.url, args.output_dir, args.timeout)
    failures = []
    base = {"model": args.model, "temperature": 0, "max_tokens": 256,
            "chat_template_kwargs": {"thinking": False}}
    france = [{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}]
    tool_user = [{"role": "user", "content": 'Call lookup once with city exactly "Paris". Do not answer without calling the tool.'}]
    tools = [{"type": "function", "function": {
        "name": "lookup", "description": "Return the current temperature for the given city.",
        "strict": True, "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}}, "required": ["city"], "additionalProperties": False},
    }}]

    def run(name, payload, validator, expected_status=200):
        try:
            message = probe.call(name, payload, expected_status=expected_status)
            validator(message)
            print(name, "PASS", flush=True)
            return message
        except Exception as error:
            failures.append({"name": name, "error": str(error)})
            print(name, "FAIL", str(error)[:500], flush=True)
            return None

    def paris(message):
        assert "paris" in (message.get("content") or "").lower(), message

    def thinking(message):
        paris(message)
        assert (message.get("reasoning_content") or "").strip(), message
        assert "<think>" not in message["content"] and "</think>" not in message["content"], message

    def lookup(message):
        calls = message.get("tool_calls") or []
        assert len(calls) == 1, message
        assert calls[0].get("id"), calls
        assert calls[0]["function"]["name"] == "lookup", calls
        assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}, calls

    for stream in (False, True):
        run(f"chat-stream-{stream}", {**base, "messages": france, "stream": stream,
            "stream_options": {"include_usage": True} if stream else None}, paris)
    run("thinking-budget-75", {**base, "messages": france, "reasoning_effort": 75,
        "chat_template_kwargs": {"thinking": True}}, thinking)
    for choice, stream in (("auto", False), ("auto", True), ("required", False),
                           ({"type": "function", "function": {"name": "lookup"}}, False)):
        name = f"tool-{choice if isinstance(choice, str) else 'named'}-stream-{stream}"
        message = run(name, {**base, "messages": tool_user, "tools": tools,
            "tool_choice": choice, "stream": stream,
            "stream_options": {"include_usage": True} if stream else None}, lookup)
        if choice == "auto" and not stream and message:
            # A synthetic fixture response, NOT an executed external tool.
            messages = tool_user + [message, {"role": "tool",
                "tool_call_id": message["tool_calls"][0]["id"],
                "content": '{"city":"Paris","temperature_c":22}'},
                {"role": "user", "content": "State the temperature from that tool result in one short sentence. Do not call any more tools."}]

            def temperature(reply):
                assert "22" in (reply.get("content") or ""), reply
                assert not reply.get("tool_calls"), reply

            run("tool-result-roundtrip", {**base, "messages": messages, "tools": tools,
                "tool_choice": "none"}, temperature)
    for effort in (101, 0.5):
        run(f"invalid-effort-{effort}", {**base, "messages": france, "reasoning_effort": effort},
            lambda _: None, expected_status=400)
    summary = {"passed": not failures, "url": args.url, "model": args.model,
               "checks": len(probe.results), "failures": failures,
               "results": [{k: r[k] for k in ("name", "status", "elapsed_s", "finish_reason", "message") if k in r}
                           for r in probe.results],
               "scope": "Text/thinking/tool API smoke; no external tools executed, no broad quality claim."}
    probe.save("summary", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
