"""Text extraction + parsing of syllabus, PYQs and study material."""
import re, pathlib

def read_text(path):
    from . import extract
    return extract.text(path)

def read_pages(path):
    """[(location, text)] - 'p. N' for PDF pages, 'slide N' for PPT/PPTX, 'para N' for DOCX/TXT/MD blocks.
    Scanned/handwritten pages and slide images are read via OCR/vision (cached; see extract.py)."""
    from . import extract
    return extract.pages(path)

# ---------------- Syllabus ----------------
# Headers: "Module 1: X", "UNIT – I", "Unit I INTRODUCTION 9", "Chapter 2 - X (8 hrs)", "MODULE-III", "Unit 1" (+ title on next line)
MOD_RE = re.compile(r"^\s*(?:module|unit|chapter|part)\s*[-\u2013\u2014:.]?\s*([0-9]+|[IVX]+)\b\s*[:.\-\u2013\u2014)]*\s*(.*)$", re.I)
HOURS = re.compile(r"[\(\[]?\s*\d+\s*(?:hrs?|hours?|lectures?|L|periods?)\s*[\)\]]?\.?$|\s+\d{1,2}\s*$", re.I)
STOP_SECTIONS = re.compile(r"^\s*(text\s*books?|reference\s*books?|references|course\s*outcomes?|suggested\s*readings?|"
                           r"list\s*of\s*experiments|web\s*resources|e-?resources|assessment|evaluation\s*scheme)\b", re.I)
NOISE = re.compile(r"^\s*(co\d|l\s*t\s*p\s*c|course\s*(code|objectives?)|prerequisites?|total\b|"
                   r"objectives?\s*:|outcomes?\s*:|credits?|semester|page\s*\d+)", re.I)
ITEM_NOISE = re.compile(r"^\s*(co\d|total\b|l\s*t\s*p\s*c|page\s*\d+)", re.I)

def _split_items(s):
    """Split on , ; and spaced dashes, outside parentheses."""
    s = re.sub(r"\s+[\u2013\u2014-]\s+", ";", s)
    out, buf, depth = [], "", 0
    for ch in s:
        depth += ch == "("; depth -= ch == ")"
        if ch in ",;" and depth == 0:
            out.append(buf); buf = ""
        else:
            buf += ch
    out.append(buf)
    return [x.strip(" .-\u2022*\t:") for x in out if x.strip(" .-\u2022*\t:")]

def _clean_title(t):
    t = HOURS.sub("", t.strip()).strip(" :.-\u2013\u2014")
    return t.title() if t.isupper() else t

ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}
SKIP_LINE = re.compile(r"^\s*(teaching\s*hours|\d+\s*(hours?|hrs?|periods?)|text\s*book\s*\d*\s*:.*|module\s*$)\s*$", re.I)

def _mod_num(tok, fallback):
    tok = tok.strip().lower()
    return int(tok) if tok.isdigit() else ROMAN.get(tok, fallback)

def _caps_prefix(line):
    """'NON-LINEAR MODELS Non-linear models: ...' -> ('NON-LINEAR MODELS', 'Non-linear models: ...')"""
    m = re.match(r"^((?:[A-Z][A-Z\-&/]+\s+){1,8}[A-Z][A-Z\-&/]+)\b\s*(.*)$", line)
    return (m.group(1), m.group(2)) if m and len(m.group(1)) >= 6 else (None, line)

def _items_with_groups(line):
    """'Multivariate Calculus: Gradient, Hessian; Linear Algebra: Determinants' -> [(group, item)]"""
    line = re.split(r"\btext\s*book\s*\d*\s*:", line, flags=re.I)[0]          # drop textbook references
    line = re.sub(r"\.\s+(?=[A-Z])", ";", line)                               # sentence breaks separate items
    out, group = [], ""
    for seg in re.split(r";", line):
        for item in _split_items(seg):
            if ":" in item:
                g, item = item.split(":", 1)
                group, item = g.strip(), item.strip()
            g_item = group
            m = re.match(r"^(\S+(?:\s+\S+)+?)-(?=[A-Z])(.+)$", item)              # 'Group phrase-Specific Item'
            if m and len(m.group(1).split()) >= 2 and not m.group(1).lower().startswith("the "):
                g_item, item = (f"{group}: {m.group(1)}" if group else m.group(1)), m.group(2)   # this item only
            if item:
                out.append((g_item, item))
        group = ""
    return out

def parse_syllabus(text):
    """Returns list of {module, topic, group, subtopic, text}."""
    rows, topic, module, pending_title, in_stop = [], None, 0, False, False
    for raw in text.replace("\xa0", " ").splitlines():
        line = raw.strip()
        if not line or SKIP_LINE.match(line):
            continue
        if STOP_SECTIONS.match(line):
            in_stop = True; continue
        m = MOD_RE.match(line)
        if m:
            in_stop = False
            module = _mod_num(m.group(1), module + 1)
            rest = _clean_title(m.group(2))
            head, _, body = rest.partition(":") if ":" in rest else (rest, "", "")
            if len(head.split()) > 8 and not body:
                topic, body = f"Module {module}", rest
            else:
                topic = head.strip() or f"Module {module}"
            pending_title = not head.strip()
            if body.strip():
                line = body
            else:
                continue
        elif in_stop or NOISE.match(line):
            continue
        elif pending_title:
            cap, rest = _caps_prefix(line)
            if cap:                                          # 'NON-LINEAR MODELS Non-linear models: ...'
                topic = _clean_title(cap); pending_title = False
                if not rest.strip():
                    continue
                line = rest
            elif len(line.split()) <= 8 and "," not in line:
                topic = _clean_title(line); pending_title = False
                continue
        pending_title = False
        line = line.strip(" -\u2022*\t")
        if topic is None:
            head, _, rest = line.partition(":")
            if rest.strip() and len(head) < 40:
                module += 1; topic = head.strip(); line = rest
            elif "," not in line:
                continue
            else:
                module += 1; topic = "General"
        for group, item in _items_with_groups(line):
            name = re.sub(r"\s*\(.*?\)", "", item).strip(" .")
            if 2 < len(name) <= 80 and len(name.split()) <= 12 and not ITEM_NOISE.match(name):
                rows.append(dict(module=module, topic=topic, group=group, subtopic=name, text=item.strip(" .")))
    seen, out = set(), []
    for r in rows:
        k = (r["module"], r["subtopic"].lower())
        if k not in seen:
            seen.add(k); out.append(r)
    return out

# ---------------- PYQs ----------------
Q_RE = re.compile(r"^\s*(?:Q(?:uestion)?\s*\.?\s*(\d{1,2})|(\d{1,2})\s*[.)])\s*[.):]?\s*(.*)$", re.I)
S_RE = re.compile(r"^\s*\(?([a-h]|i{1,3}|iv|vi?)\)\s*(.*)$", re.I)
M_RE = re.compile(r"\s*(?:[\[(]\s*(\d{1,2})\s*(?:m|marks?)?\s*[\])]|(\d{1,2})\s*marks?)\s*$", re.I)

MONTHS = dict(jan=1, feb=2, mar=3, apr=4, may=5, jun=6, jul=7, aug=8, sep=9, oct=10, nov=11, dec=12)

def year_from_name(name):
    m = re.search(r"(19|20)\d{2}", name)
    return int(m.group()) if m else None

def paper_date(text, name=""):
    """Exam year from the paper itself ('04-Dec-2023', 'NOV/DEC 2023'), else from the file name."""
    m = re.search(r"\b\d{1,2}[-/ ]([A-Za-z]{3})[A-Za-z]*[-/ ]((?:19|20)\d{2})\b", text)
    if m and m.group(1).lower()[:3] in MONTHS:
        return int(m.group(2)), MONTHS[m.group(1).lower()[:3]]
    m = re.search(r"\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*(?:\s*/\s*[A-Z]+)?\s*[,-]?\s*((?:19|20)\d{2})\b", text, re.I)
    if m:
        return int(m.group(1)), None
    return year_from_name(name), None

Q_LINE = re.compile(r"^\s*(?:Q(?:uestion)?\s*\.?\s*(\d{1,2})|(\d{1,2})\s*[.)])\s*[.):]?\s*(.*)$", re.I)
PART = re.compile(r"^\s*\(?([A-Ha-h]|i{1,3}|iv|vi?)\)\s*(.*)$")
MARKS = re.compile(r"[\(\[]\s*(\d{1,2})\s*(?:m|marks?)?\s*[\)\]]\s*$|(\d{1,2})\s*marks?\s*$", re.I)

def parse_pyq(text, paper_id, year):
    """Handles '1) text (4) / A) / B) text (3) / a) sub-part' layouts and 'Q1 (a) ... [8]' layouts.
    Lowercase a)/b) inside an uppercase part stay part of that question; table rows are not mistaken
    for question numbers (numbers must run in sequence)."""
    qs, cur, last_q = [], None, 0
    def flush():
        if cur and cur["question"].strip():
            qtext = re.sub(r"\s+", " ", cur["question"]).strip()
            if len(qtext.split()) >= 3:
                qs.append({k: v for k, v in dict(cur, question=qtext).items() if not k.startswith("_")})
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        mk = MARKS.search(line)
        if mk:
            line = MARKS.sub("", line).strip()
        qm = Q_LINE.match(line)
        if qm:
            n = int(qm.group(1) or qm.group(2)); rest = qm.group(3)
            if n == last_q + 1 and (not rest or rest[0].isalpha() or rest[0] in "([") :
                flush(); last_q = n
                sm = PART.match(rest)
                cur = dict(paper_id=paper_id, year=year, qno=str(n), part=sm.group(1).upper() if sm else "",
                           question=sm.group(2) if sm else rest, marks=None,
                           _upper=bool(sm and sm.group(1).isupper()))
                if mk and cur["marks"] is None:
                    cur["marks"] = int(mk.group(1) or mk.group(2))
                continue
        sm = PART.match(line)
        if sm and cur is not None:
            letter, rest = sm.group(1), sm.group(2)
            if letter.isupper() or not cur["part"] or not cur.get("_upper"):
                if not cur["part"] and letter.upper() == "A":
                    cur["part"] = "A"; cur["_upper"] = letter.isupper()   # 'A)' labels/continues the text of this question
                    cur["question"] += " " + rest
                else:
                    flush()
                    cur = dict(paper_id=paper_id, year=year, qno=cur["qno"], part=letter.upper(), question=rest,
                               marks=None, _upper=letter.isupper())
                if mk and cur["marks"] is None:
                    cur["marks"] = int(mk.group(1) or mk.group(2))
                continue
        if cur is not None:
            if re.fullmatch(r"[\d\s.:|,_-]+", line) or re.match(r"(?i)^-+\s*end\s*-+$|^page \d+ of \d+$", line):
                continue                                     # table rows of numbers, footers
            cur["question"] += " " + line
            if mk and cur["marks"] is None:
                cur["marks"] = int(mk.group(1) or mk.group(2))
    flush()
    return qs

def chunk_material(text, size=80):
    """Split study material into ~size-word chunks."""
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size // 2) if words[i:i + size]]
