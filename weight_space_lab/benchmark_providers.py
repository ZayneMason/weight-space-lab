"""Minimal native-tool API adapters. Secrets are headers only; no retries or redirects."""
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .editor_session import _strict_json

KEYS = {"anthropic": "ANTHROPIC_API_KEY", "fireworks": "FIREWORKS_API_KEY"}
ENDPOINTS = {"anthropic": "https://api.anthropic.com/v1/messages",
             "fireworks": "https://api.fireworks.ai/inference/v1/chat/completions"}


def tools():
    def schema(properties=None, required=None):
        return {"type": "object", "properties": properties or {}, "required": required or [], "additionalProperties": False}
    text = {"type": "string"}
    candidate = schema({"candidate_id": text}, ["candidate_id"])
    patch = schema({"base_sha256": text, "reason": text,
        "operations": {"type": "array", "items": schema({"op": {"type": "string", "enum": ["set", "add", "scale"]},
            "tensor": text, "indices": {"type": "array", "items": {"type": "integer"}},
            "values": {"type": "array", "items": {"type": "number"}}}, ["op", "tensor", "indices", "values"])}},
        ["base_sha256", "reason", "operations"])
    descriptions = {
        "describe": ("Read the task, editable tensor/indices, budgets and success rule before editing.", schema()),
        "inspect": ("Read original checkpoint SHA-256, tensor names/shapes/statistics, and the dense-layer architecture.", schema()),
        "tensor": ("Read original tensor values by name with a bounded flat slice.", schema({"name": text, "start": {"type": "integer"}, "count": {"type": "integer"}}, ["name"])),
        "practice_inputs": ("Read unlabeled practice input vectors; labels require metered feedback.", schema()),
        "propose": ("Submit an inline patch against the original checkpoint hash. Set replaces selected values; add adds; scale multiplies. Proposals, including rejections, consume budget. All candidates start from the original.", schema({"patch": patch}, ["patch"])),
        "feedback": ("Spend one practice feedback call on an accepted candidate. Returns labels/predictions, fixes/regressions, and matched baseline accuracy.", candidate),
        "freeze": ("Select an accepted candidate with completed feedback. Ends editing permanently, before separate operator held-out evaluation.", candidate),
        "abstain": ("Close the trial without selecting a candidate. Records an unsuccessful selection outcome and ends editing.", schema({"reason": text}, ["reason"])),
        "status": ("Read phase and remaining global proposal/feedback allowances.", schema())}
    return [{"name": name, "description": description, "input_schema": parameters}
            for name, (description, parameters) in descriptions.items()]


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def send(provider, body, timeout=90):
    key = os.environ.get(KEYS[provider])
    if not key:
        raise ValueError(f"Missing {KEYS[provider]}")
    headers = {"Content-Type": "application/json"}
    if provider == "anthropic":
        headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
    else:
        headers["Authorization"] = "Bearer " + key
    request = Request(ENDPOINTS[provider], json.dumps(body, allow_nan=False).encode(), headers)
    try:
        with build_opener(NoRedirect).open(request, timeout=timeout) as response:
            data = response.read(4*1024*1024 + 1)
            if len(data) > 4*1024*1024:
                raise ValueError("Provider response exceeds the recorded limit")
            return json.loads(data)
    except HTTPError as exc:
        raise RuntimeError(f"Provider HTTP status {exc.code}; no retry was attempted") from None
    except (URLError, TimeoutError, OSError):
        raise RuntimeError("Provider transport failed; billing is uncertain and no retry was attempted") from None


class Conversation:
    def __init__(self, model, prompt):
        self.model = model
        self.provider = model["provider"]
        self.prompt = prompt
        self.messages = ([{"role": "user", "content": prompt}] if self.provider == "anthropic"
                         else [{"role": "system", "content": prompt}, {"role": "user", "content": "Begin the bounded editing task."}])

    def request(self, max_output_tokens):
        body = {"model": self.model["model"], "messages": self.messages, "max_tokens": max_output_tokens}
        if "temperature" in self.model["settings"]:
            body["temperature"] = self.model["settings"]["temperature"]
        native = tools()
        body["tools"] = native if self.provider == "anthropic" else [
            {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in native]
        return body

    def receive(self, raw):
        if self.provider == "anthropic":
            self.messages.append({"role": "assistant", "content": raw["content"]})
            calls = [{"id": b["id"], "name": b["name"], "params": b["input"]}
                     for b in raw["content"] if b["type"] == "tool_use"]
        else:
            message = raw["choices"][0]["message"]
            self.messages.append({k: v for k, v in message.items() if k in ("role", "content", "tool_calls", "reasoning_content")})
            calls = []
            for call in message.get("tool_calls") or []:
                args = call["function"]["arguments"]
                try:
                    params = _strict_json(args)
                except (ValueError, TypeError, RecursionError):
                    params = args  # Retain malformed arguments; the bridge records/rejects them.
                calls.append({"id": call["id"], "name": call["function"]["name"], "params": params})
        return calls, usage(self.provider, raw)

    def results(self, replies):
        if self.provider == "anthropic":
            self.messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": call_id, "content": json.dumps(result), "is_error": "error" in result}
                for call_id, result in replies]})
        else:
            self.messages.extend({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)} for call_id, result in replies)


def usage(provider, raw):
    """Account reported tokens independently of tool parsing, including failed turns."""
    reported = raw["usage"]
    if provider == "anthropic":
        inputs = [reported["input_tokens"], reported.get("cache_creation_input_tokens", 0), reported.get("cache_read_input_tokens", 0)]
        output = reported["output_tokens"]
    else:
        inputs, output = [reported["prompt_tokens"]], reported["completion_tokens"]
    if any(type(v) is not int or v < 0 for v in [*inputs, output]):
        raise ValueError("Provider returned invalid usage accounting")
    return {"input_tokens": sum(inputs), "output_tokens": output}
