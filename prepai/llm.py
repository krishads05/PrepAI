"""LLM wrapper - optional. Backends (auto-detected, in order):
  1. Ollama (free, local):  install from ollama.com, then `ollama pull qwen2.5:3b`
  2. Anthropic API:         set ANTHROPIC_API_KEY
Force one with PREPAI_LLM=ollama|anthropic|none. Without any, PrepAI runs fully offline."""
import os, json, re, urllib.request
OLLAMA = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("PREPAI_OLLAMA_MODEL", "qwen2.5:3b")
ANTHROPIC_MODEL = os.getenv("PREPAI_MODEL", "claude-haiku-4-5-20251001")
_backend = None

def _ollama_up():
    """True if Ollama is running with a model. Uses PREPAI_OLLAMA_MODEL if pulled, else the first installed model."""
    global OLLAMA_MODEL
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=1) as r:
            names = [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return False
    if not names:
        return False
    if OLLAMA_MODEL not in names and f"{OLLAMA_MODEL}:latest" not in names:
        OLLAMA_MODEL = names[0]
    return True

def reset():
    """Re-detect the backend (e.g. after starting Ollama while the app is running)."""
    global _backend
    _backend = None
    return backend()

def model_name():
    return OLLAMA_MODEL if backend() == "ollama" else ANTHROPIC_MODEL if backend() == "anthropic" else "-"

def backend():
    global _backend
    if _backend is None:
        forced = os.getenv("PREPAI_LLM")
        if forced:
            _backend = forced
        elif _ollama_up():
            _backend = "ollama"
        elif os.getenv("ANTHROPIC_API_KEY"):
            _backend = "anthropic"
        else:
            _backend = "none"
    return _backend

def available():
    return backend() != "none"

def ask(prompt, max_tokens=1500, system=None, json_mode=False):
    if backend() == "ollama":
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = dict(model=OLLAMA_MODEL, messages=msgs, stream=False, options={"num_predict": max_tokens, "temperature": 0.3})
        if json_mode:
            body["format"] = "json"
        req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)["message"]["content"]
    import anthropic
    kw = dict(model=ANTHROPIC_MODEL, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
    if system:
        kw["system"] = system
    r = anthropic.Anthropic().messages.create(**kw)
    return "".join(b.text for b in r.content if b.type == "text")

def ask_json(prompt, **kw):
    txt = ask(prompt + "\nRespond with ONLY valid JSON, no markdown fences.", json_mode=True, **kw)
    txt = re.sub(r"```(?:json)?|```", "", txt).strip()
    return json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
