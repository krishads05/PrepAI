"""Show past-paper questions as printed: crop each question's region (tables, figures, diagrams) from the
original PDF page, and strip flattened table text from the displayed question.
Works for text PDFs (word positions) and scanned PDFs (OCR line boxes)."""
import difflib, io, pathlib, re

DPI = 150
VISUAL_RE = re.compile(r"\b(table|fig\.?|figure|diagram|graph|plot|network|dendrogram|tree shown|"
                       r"(?:data|dataset|matrix) (?:shown|given|listed|below)|following (?:data|table|figure|matrix))\b", re.I)

def _norm(t):
    return " ".join(re.findall(r"[a-z0-9]+", t.lower()))

def _lines_text_pdf(path):
    """[(page, y0, y1, text)] in pixel coords at DPI, from the PDF's own text layer."""
    import pdfplumber
    out, k = [], DPI / 72
    with pdfplumber.open(path) as d:
        for p, page in enumerate(d.pages):
            rows = {}
            for w in page.extract_words(keep_blank_chars=False, use_text_flow=True):
                key = round(w["top"] / 3)
                rows.setdefault(key, []).append(w)
            for key in sorted(rows):
                ws = sorted(rows[key], key=lambda w: w["x0"])
                out.append((p, min(w["top"] for w in ws) * k, max(w["bottom"] for w in ws) * k, " ".join(w["text"] for w in ws)))
    return out

def _lines_ocr(img):
    """[(y0, y1, text)] line boxes via RapidOCR or Tesseract."""
    from . import vision
    if vision.rapidocr_available():
        import numpy as np
        from PIL import Image
        vision._rapidocr(img)                               # initialise engine
        arr = np.array(Image.open(io.BytesIO(img)).convert("RGB"))[:, :, ::-1]
        res = vision._rapid(arr)
        items = [(b, t) for b, t, _ in (res[0] or [])] if isinstance(res, tuple) else list(zip(res.boxes, res.txts))
        return sorted((min(p[1] for p in b), max(p[1] for p in b), t) for b, t in items)
    exe = vision.tesseract_path()
    if not exe:
        return []
    import subprocess, tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(img); path = f.name
    try:
        tsv = subprocess.run([exe, path, "-", "--psm", "6", "tsv"], capture_output=True, text=True, timeout=120).stdout
    finally:
        os.unlink(path)
    lines = {}
    for row in tsv.splitlines()[1:]:
        c = row.split("\t")
        if len(c) == 12 and c[11].strip():
            key = (c[2], c[3], c[4]); y, h = int(c[7]), int(c[9])
            a = lines.setdefault(key, [y, y + h, []]); a[0] = min(a[0], y); a[1] = max(a[1], y + h); a[2].append(c[11])
    return sorted((a[0], a[1], " ".join(a[2])) for a in lines.values())

def page_lines(path):
    from . import extract
    lines = []
    texts = extract._pdf_text_pages(path)
    native = _lines_text_pdf(path) if any(len(t.strip()) >= 40 for t in texts) else []
    for p, t in enumerate(texts):
        if len(t.strip()) >= 40:
            lines += [l for l in native if l[0] == p]
        else:
            lines += [(p, y0, y1, txt) for y0, y1, txt in _lines_ocr(extract._render_page(path, p, dpi=DPI))]
    return lines

def _locate(qs, lines):
    """Index into `lines` where each question starts (document order), or None."""
    norm = [_norm(l[3]) for l in lines]
    pos, start = [], 0
    for q in qs:
        key = _norm(q["question"])[:60]
        best, best_i = 0.0, None
        for i in range(start, len(lines)):
            cand = _norm(" ".join(l[3] for l in lines[i:i + 2]))
            r = difflib.SequenceMatcher(None, key[:45], cand[:60]).ratio() if cand else 0
            if r > best:
                best, best_i = r, i
            if r > 0.85:
                break
        if best_i is not None and best >= 0.55:
            pos.append(best_i); start = best_i + 1
        else:
            pos.append(None)
    return pos

def attach_images(pdf, qs, out_dir, only_visual=True):
    """Crop each question's printed region; sets q['image'] (path) where the question needs a visual."""
    from . import extract
    from PIL import Image
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    lines = page_lines(pdf)
    if not lines or not qs:
        return qs
    pos = _locate(qs, lines)
    pages = {}
    def page_img(p):
        if p not in pages:
            pages[p] = Image.open(io.BytesIO(extract._render_page(pdf, p, dpi=DPI))).convert("RGB")
        return pages[p]
    nxt = [next((pos[j] for j in range(i + 1, len(pos)) if pos[j] is not None), None) for i in range(len(pos))]
    for i, q in enumerate(qs):
        if pos[i] is None or (only_visual and not needs_visual(q["question"])):
            continue
        s = lines[pos[i]]
        e = lines[nxt[i]] if nxt[i] is not None else None
        p0 = s[0]; im0 = page_img(p0); W, H = im0.size
        y_start = max(0, s[1] - 10)
        if e is not None and e[0] == p0:
            parts = [im0.crop((int(W * .03), int(y_start), int(W * .97), int(e[1] - 3)))]
        else:
            parts = [im0.crop((int(W * .03), int(y_start), int(W * .97), int(H * .93)))]
            if e is not None and e[0] == p0 + 1:
                im1 = page_img(e[0]); W1, H1 = im1.size
                top = int(H1 * .04)
                if e[1] - top > 30:
                    parts.append(im1.crop((int(W1 * .03), top, int(W1 * .97), int(e[1] - 6))))
        parts = [_trim(p) for p in parts]
        parts = [p for p in parts if p is not None and p.size[1] > 20]
        if not parts:
            continue
        canvas = Image.new("RGB", (max(p.size[0] for p in parts), sum(p.size[1] for p in parts)), "white")
        y = 0
        for p in parts:
            canvas.paste(p, (0, y)); y += p.size[1]
        name = f"{pathlib.Path(pdf).stem}_Q{q['qno']}{q.get('part') or ''}.png"
        canvas.save(out_dir / name)
        q["image"] = str(out_dir / name)
    return qs

def _trim(im, pad=8):
    """Cut blank bands above/below the content (e.g. the empty bottom of a page)."""
    from PIL import ImageChops, Image
    diff = ImageChops.difference(im, Image.new("RGB", im.size, "white")).convert("L").point(lambda v: 255 if v > 40 else 0)
    box = diff.getbbox()
    if not box:
        return None
    return im.crop((0, max(0, box[1] - pad), im.size[0], min(im.size[1], box[3] + pad)))

def needs_visual(text):
    return bool(VISUAL_RE.search(text)) or _has_table_dump(text)

_TOK = re.compile(r"^(?:[A-Za-z]{1,2}|-?\d+(?:\.\d+)?%?|[|=+\-*/(),:;]+)$")
def _has_table_dump(text, run=8):
    k = 0
    for t in text.split():
        k = k + 1 if _TOK.match(t) else 0
        if k >= run:
            return True
    return False

def clean_question(text, run=6):
    """Remove flattened table cells ('a b c K 1 0 1 1 1 ...') from the displayed question."""
    toks, out, buf = text.split(), [], []
    for t in toks + [""]:
        if t and _TOK.match(t):
            buf.append(t); continue
        out += (["[…]"] if len(buf) >= run else buf); buf = []
        if t:
            out.append(t)
    return re.sub(r"(\[…\]\s*)+", "[…] ", " ".join(out)).strip()
