"""Read text inside slide images (fast) and, optionally, handwritten/scanned pages (slow, vision model).
   python process_materials.py library                 # quick: slide images only (text + equations)
   python process_materials.py library --handwritten   # also scanned/handwritten pages (needs vision model)
   python process_materials.py library --vision        # use the vision model for images too (slow)
Results are cached: stop any time and resume later."""
import sys, time
from prepai import extract, vision

args = [a for a in sys.argv[1:] if not a.startswith("--")]
folder = args[0] if args else "library"
hand, use_vision = "--handwritten" in sys.argv, "--vision" in sys.argv or "--handwritten" in sys.argv
e = vision.engines()
print(f"Text OCR: {'RapidOCR' if e['rapidocr'] else 'Tesseract' if e['tesseract'] else 'NONE (pip install rapidocr_onnxruntime)'} | "
      f"Equation OCR: {e['math_ocr'] or 'no (pip install rapid_latex_ocr)'} | Vision model: {e['vision_model'] or '-'}")
if use_vision and not e["vision_model"]:
    sys.exit("No vision model found. Run:  ollama pull qwen2.5vl:3b")
todo = extract.pending_report(folder)
print(f"Unread: {len(todo['images'])} slide images, {len(todo['pages'])} scanned/handwritten pages"
      + ("" if hand else " (skipped; add --handwritten to include)"))
t0 = time.time()
def progress(i, n, label):
    el = time.time() - t0
    print(f"\r  {i}/{n}  {label[:45]:45}  {el / 60:4.1f} min  (~{el / i * (n - i) / 60:4.1f} min left)", end="", flush=True)
r = extract.process_folder(folder, progress, include_scanned=hand, engine="vision" if use_vision else "fast")
print(f"\nDone: {r['done']} read. Remaining: {r['images_left']} images, {r['pages_left']} pages.")
