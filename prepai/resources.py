"""YouTube resource recommendation.
Mode 1 (no key): tailored YouTube search links.  Mode 2 (YOUTUBE_API_KEY set): ranked real videos via
YouTube Data API v3 (search.list + videos.list), cached in <out>/videos.json to save quota."""
import json, math, os, re, urllib.parse, urllib.request, pathlib

DEFAULT_CHANNELS = ["Gate Smashers", "Neso Academy", "Jenny's Lectures CS IT"]
API = "https://www.googleapis.com/youtube/v3"
GENERIC = {"and", "of", "the", "problems", "problem", "methods", "method", "algorithms", "algorithm", "structure"}

def queries(topic, sub, question_types="", course=""):
    """Build search queries tailored to how the topic is examined."""
    base = f"{sub} {course or topic}".strip()
    qs = [("Concept explanation", f"{base} explained")]
    if any(t in str(question_types) for t in ("numerical", "tracing")):
        qs.append(("Solved numericals", f"{sub} numerical example solved"))
    qs.append(("Quick revision", f"{sub} {course or topic} one shot revision"))
    return qs

def search_link(q, channel=None):
    return "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(f"{q} {channel}" if channel else q)

def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.load(r)

def _minutes(iso):
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    return (int(m.group(1) or 0) * 60 + int(m.group(2) or 0) + int(m.group(3) or 0) / 60) if m else 0

def api_videos(q, key, sub, channels=(), n=3):
    s = _get(f"{API}/search?" + urllib.parse.urlencode(dict(part="snippet", type="video", maxResults=10, q=q, key=key,
                                                            relevanceLanguage="en", safeSearch="strict")))
    items = s.get("items", [])
    ids = [i["id"]["videoId"] for i in items]
    if not ids:
        return []
    v = {x["id"]: x for x in _get(f"{API}/videos?" + urllib.parse.urlencode(
        dict(part="contentDetails,statistics", id=",".join(ids), key=key))).get("items", [])}
    words = {w for w in re.findall(r"[a-z0-9']+", sub.lower()) if w not in GENERIC and len(w) > 2}
    out = []
    for rank, it in enumerate(items):
        vid, sn = it["id"]["videoId"], it["snippet"]
        det = v.get(vid, {})
        mins = _minutes(det.get("contentDetails", {}).get("duration"))
        views = int(det.get("statistics", {}).get("viewCount", 0) or 0)
        if mins < 3:                                            # skip Shorts / clips
            continue
        title = sn["title"]
        rel = len(words & set(re.findall(r"[a-z0-9']+", title.lower()))) / max(len(words), 1)
        if rel == 0:                                            # title doesn't mention the topic -> drop
            continue
        fit = 1.0 if 5 <= mins <= 35 else 0.5 if mins <= 70 else 0.2
        pref = 1.0 if any(c.lower() in sn["channelTitle"].lower() for c in channels) else 0.0
        score = 0.30 * (1 - rank / len(items)) + 0.25 * min(math.log10(views + 1) / 6, 1) + 0.25 * rel + 0.10 * fit + 0.10 * pref
        out.append(dict(title=title, channel=sn["channelTitle"], url=f"https://www.youtube.com/watch?v={vid}",
                        minutes=round(mins), views=views, score=round(score, 3)))
    return sorted(out, key=lambda x: -x["score"])[:n]

def recommend(topic, sub, question_types="", channels=DEFAULT_CHANNELS, course="", out_dir=None, key=None):
    """Returns {'mode': 'api'|'links', 'items': [...]} for one subtopic."""
    key = key or os.getenv("YOUTUBE_API_KEY")
    if key:
        cache_p = pathlib.Path(out_dir) / "videos.json" if out_dir else None
        cache = json.loads(cache_p.read_text()) if cache_p and cache_p.exists() else {}
        ck = f"{sub}|{','.join(channels)}"
        if ck not in cache:
            items = []
            for label, q in queries(topic, sub, question_types, course)[:2]:     # 2 searches/topic = 200 quota units
                for vdict in api_videos(q, key, sub, channels):
                    if vdict["url"] not in {i["url"] for i in items}:
                        items.append(dict(vdict, purpose=label))
            cache[ck] = items
            if cache_p:
                cache_p.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        return dict(mode="api", items=cache[ck])
    items = []
    for label, q in queries(topic, sub, question_types, course):
        items.append(dict(purpose=label, title=q, url=search_link(q)))
    for ch in channels[:3]:
        items.append(dict(purpose=f"On {ch}", title=f"{sub} - {ch}", url=search_link(sub, ch)))
    return dict(mode="links", items=items)
