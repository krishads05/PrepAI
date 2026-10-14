"""Vision/OCR backends.
  1. Local vision LLM via Ollama (reads handwriting, equations, diagrams) - e.g. `ollama pull qwen2.5vl:3b`
  2. Tesseract OCR (printed text only)
Override the model with PREPAI_VISION_MODEL."""
import base64, io, json, os, re, shutil, subprocess, tempfile, urllib.request
OLLAMA = os.getenv("OLLAMA_HOST", "http://localhost:11434")
VISION_RE = re.compile(r"(vl\b|vl:|vl$|vision|llava|minicpm-v|moondream|gemma3|bakllava|qwen3-vl|qwen2\.5vl)", re.I)
_model = None

PROMPTS = {
    "page": ("This is a page of a student's study notes or an exam paper; it may be handwritten. Transcribe ALL text "
             "exactly, line by line. Write math in plain text (x^2, d/dx, sum, sqrt, matrices as [[a,b],[c,d]]). "
             "If there is a diagram or table, add a line starting 'Diagram:' or 'Table:' describing it. "
             "Output only the transcription."),
    "image": ("This image comes from a lecture slide. Transcribe any text and equations exactly (math in plain text). "
              "If it is a diagram, chart or figure, describe what it shows in 1-3 sentences with its key labels. "
              "Output only the transcription/description."),
}

def vision_model():
    """Installed Ollama vision model name, or None."""
    global _model
    if _model is None:
        _model = os.getenv("PREPAI_VISION_MODEL", "")
        if not _model:
            try:
                with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=2) as r:
                    names = [m["name"] for m in json.load(r).get("models", [])]
                _model = next((n for n in names if VISION_RE.search(n)), "")
            except Exception:
                _model = ""
    return _model or None

def reset():
    global _model
    _model = None
    return vision_model()

def tesseract_path():
    p = shutil.which("tesseract")
    if p:
        return p
    for c in (r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if os.path.exists(c):
            return c
    return None

def engines():
    return dict(vision_model=vision_model(), tesseract=bool(tesseract_path()), rapidocr=rapidocr_available(),
                math_ocr=math_ocr_available())

def _shrink(img, max_side=1280):
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(img)).convert("RGB")
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side))
        buf = io.BytesIO(); im.save(buf, "PNG"); return buf.getvalue()
    except Exception:
        return img

def _vlm(img, kind, model):
    body = dict(model=model, stream=False, options={"temperature": 0, "num_predict": 1500},
                messages=[{"role": "user", "content": PROMPTS[kind], "images": [base64.b64encode(_shrink(img)).decode()]}])
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["message"]["content"].strip()

def _lines_from_words(words, min_conf=70):
    """words: [(line_key, text, conf)] -> confident words only, regrouped into lines; junk lines dropped."""
    lines, cur, buf = [], None, []
    for key, w, c in words:
        if key != cur and buf:
            lines.append(buf); buf = []
        cur = key
        if c >= min_conf and (re.search(r"[A-Za-z]{2,}", w) or re.fullmatch(r"[\d.,=+\-<>%()]+", w)):
            buf.append(w)
    if buf:
        lines.append(buf)
    good = [" ".join(l) for l in lines if sum(bool(re.search(r"[A-Za-z]{2,}", w)) for w in l) >= 1]
    return "\n".join(good)

def _tesseract(img):
    """Returns (text_of_confident_words, mean_confidence)."""
    exe = tesseract_path()
    if not exe:
        return "", 0.0
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(_shrink(img, 2400)); path = f.name
    try:
        out = subprocess.run([exe, path, "-", "--psm", "6", "tsv"], capture_output=True, text=True, timeout=120).stdout
    finally:
        os.unlink(path)
    words = []
    for row in out.splitlines()[1:]:
        c = row.split("\t")
        if len(c) == 12 and c[11].strip() and float(c[10]) >= 0:
            words.append(((c[2], c[3], c[4]), c[11], float(c[10])))
    conf = sum(w[2] for w in words) / len(words) if words else 0.0
    return _lines_from_words(words), conf

_rapid = None
def _rapidocr(img):
    """RapidOCR (pip install rapidocr_onnxruntime) - pure Python, no system install. Returns (text, conf)."""
    global _rapid
    import numpy as np
    from PIL import Image
    if _rapid is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            from rapidocr import RapidOCR
        _rapid = RapidOCR()
    arr = np.array(Image.open(io.BytesIO(img)).convert("RGB"))[:, :, ::-1]
    res = _rapid(arr)
    if isinstance(res, tuple):                     # rapidocr_onnxruntime: (result, elapse)
        items = [(b, t, float(sc)) for b, t, sc in (res[0] or [])]
    else:                                          # rapidocr >= 2: result object
        items = list(zip(res.boxes if res.boxes is not None else [], res.txts or [], [float(x) for x in (res.scores or [])]))
    items.sort(key=lambda x: (round(min(p[1] for p in x[0]) / 15), min(p[0] for p in x[0])))
    words = [(round(min(p[1] for p in b) / 15), t, sc * 100) for b, t, sc in items]
    conf = sum(w[2] for w in words) / len(words) if words else 0.0
    return _lines_from_words(words, min_conf=60), conf

def rapidocr_available():
    try:
        import rapidocr_onnxruntime  # noqa
        return True
    except ImportError:
        try:
            import rapidocr  # noqa
            return True
        except ImportError:
            return False

_latex = None
def math_ocr_available():
    try:
        import rapid_latex_ocr  # noqa
        return True
    except ImportError:
        return False

def math_ocr(img):
    """Equation image -> LaTeX (pip install rapid_latex_ocr). Returns '' if unavailable or not an equation."""
    global _latex
    if not math_ocr_available():
        return ""
    try:
        if _latex is None:
            from rapid_latex_ocr import LaTeXOCR
            _latex = LaTeXOCR()
        out = _latex(img)
        tex = (out[0] if isinstance(out, tuple) else out) or ""
        tex = str(tex).strip()
        return tex if len(tex) >= 3 and re.search(r"[=\\^_+]", tex) else ""
    except Exception as e:
        print(f"[WARN] equation OCR failed ({type(e).__name__})")
        return ""

def fast_ocr(img):
    """Fast OCR: words via RapidOCR/Tesseract; if an image has almost no words, try equation OCR."""
    text, eng = _fast_words(img)
    if len(text.split()) < 3:
        tex = math_ocr(img)
        if tex:
            return (f"{text}\nEquation: {tex}".strip(), f"{eng}+latex")
    return text, eng

def _fast_words(img):
    """Fast printed-text OCR: RapidOCR if installed, else Tesseract. Returns (text, engine)."""
    if rapidocr_available():
        try:
            t, c = _rapidocr(img); return t, f"rapidocr({c:.0f})"
        except Exception as e:
            print(f"[WARN] RapidOCR failed ({type(e).__name__}); trying Tesseract")
    if tesseract_path():
        t, c = _tesseract(img); return t, f"tesseract({c:.0f})"
    return "", "none"

def ocr(img, kind="page", engine="auto"):
    """Returns (text, engine_label).
    engine='fast'  : RapidOCR/Tesseract only (seconds per image; printed text, labels, titles)
    engine='vision': local vision model (slow; handwriting, equations, diagram descriptions)
    engine='auto'  : vision model if installed, else fast OCR."""
    if engine in ("vision", "auto"):
        m = vision_model()
        if m:
            try:
                return _vlm(img, kind, m), f"vision:{m}"
            except Exception as e:
                print(f"[WARN] vision model failed ({type(e).__name__}); using fast OCR")
        elif engine == "vision":
            return "", "none"
    return fast_ocr(img)
