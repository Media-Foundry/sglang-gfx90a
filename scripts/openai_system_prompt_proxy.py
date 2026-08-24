#!/usr/bin/env python3
"""OpenAI-compatible reverse proxy that enforces a server-side system prompt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask


HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
PROXY_GENERATED_RESPONSE_HEADERS = HOP_BY_HOP_HEADERS | {"date", "server"}


def inject_system_prompt(payload: dict[str, Any], system_prompt: str) -> None:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("chat/completions payload must contain a messages list")

    # Append to the last leading string-valued system message. This preserves
    # the client's prompt while ensuring the server policy has the final word
    # in the system-message prefix consumed by normal chat templates.
    leading_system_end = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "system":
            break
        leading_system_end += 1

    for index in range(leading_system_end - 1, -1, -1):
        content = messages[index].get("content")
        if isinstance(content, str):
            messages[index]["content"] = f"{content}\n\n{system_prompt}"
            return

    messages.insert(
        leading_system_end,
        {"role": "system", "content": system_prompt},
    )


def create_app(upstream: str, system_prompt_file: Path) -> FastAPI:
    app = FastAPI(title="SGLang forced-system-prompt proxy")
    # This is always a local hop. Do not inherit HTTP(S)_PROXY from the host,
    # otherwise even 127.0.0.1 may be sent to a corporate proxy and return 502.
    client = httpx.AsyncClient(timeout=None, trust_env=False)

    @app.on_event("shutdown")
    async def close_client() -> None:
        await client.aclose()

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def proxy(path: str, request: Request):
        body = await request.body()
        if request.method == "POST" and request.url.path == "/v1/chat/completions":
            try:
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                system_prompt = system_prompt_file.read_text(encoding="utf-8").strip()
                if not system_prompt:
                    raise ValueError("forced system prompt file is empty")
                inject_system_prompt(payload, system_prompt)
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                return JSONResponse(status_code=400, content={"error": str(exc)})

        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS
        }
        upstream_path = request.url.path
        if request.method == "GET" and upstream_path == "/v1/chat/completions/models":
            upstream_path = "/v1/models"
        url = f"{upstream.rstrip('/')}{upstream_path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        upstream_request = client.build_request(
            request.method,
            url,
            headers=headers,
            content=body,
        )
        try:
            response = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            return JSONResponse(status_code=502, content={"error": str(exc)})

        response_headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() not in PROXY_GENERATED_RESPONSE_HEADERS
        }
        response_headers["x-sglang-forced-system-prompt"] = "1"
        return StreamingResponse(
            response.aiter_raw(),
            status_code=response.status_code,
            headers=response_headers,
            background=BackgroundTask(response.aclose),
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30002)
    parser.add_argument("--upstream", default="http://127.0.0.1:30001")
    parser.add_argument("--system-prompt-file", type=Path, required=True)
    args = parser.parse_args()
    uvicorn.run(
        create_app(args.upstream, args.system_prompt_file.resolve()),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
