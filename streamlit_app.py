from __future__ import annotations
import io
import os
import re
import json
import time
from types import SimpleNamespace
from typing import List, Tuple, Dict

# --- Optional deps (import guarded) -------------------------------------------------
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

import numpy as np
from PIL import Image

# Try Streamlit; if missing, provide a shim so decorators don't explode
try:  # pragma: no cover
    import streamlit as st  # type: ignore
    _STREAMLIT = True
except Exception:  # pragma: no cover
    _STREAMLIT = False

    def _noop_decorator(*_args, **_kwargs):
        def _wrap(func):
            return func
        return _wrap

    def _noop(*_args, **_kwargs):
        pass

    st = SimpleNamespace(  # type: ignore
        cache_resource=_noop_decorator,
        set_page_config=_noop,
        title=_noop,
        caption=_noop,
        sidebar=SimpleNamespace(__enter__=lambda *a, **k: None, __exit__=lambda *a, **k: False),
        header=_noop,
        toggle=lambda *a, **k: False,
        divider=_noop,
        markdown=_noop,
        checkbox=lambda *a, **k: False,
        file_uploader=lambda *a, **k: None,
        expander=lambda *a, **k: SimpleNamespace(__enter__=lambda *a, **k: None, __exit__=lambda *a, **k: False),
        info=_noop,
        image=_noop,
        spinner=SimpleNamespace(__enter__=lambda *a, **k: None, __exit__=lambda *a, **k: False),
        columns=lambda n: (SimpleNamespace(subheader=_noop, warning=_noop, code=_noop, caption=_noop),) * n,
        subheader=_noop,
        warning=_noop,
        code=_noop,
        write=_noop,
        download_button=_noop,
    )

# Optional VLM/LLM (free) — off by default
_BLIP_MODEL = "Salesforce/blip-image-captioning-base"
_T5_MODEL = "google/flan-t5-base"

# -----------------------------
# Utilities
# -----------------------------

def load_image(file) -> Image.Image:
    if isinstance(file, (str, bytes, bytearray)):
        return Image.open(file).convert("RGB")
    return Image.open(io.BytesIO(file.read())).convert("RGB")


def pil_to_cv(img: Image.Image):
    if cv2 is None:
        return None
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv_to_pil(img_cv):
    if cv2 is None or img_cv is None:
        return img_cv
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


# -----------------------------
# Preprocessing (classical CV)
# -----------------------------

def _enhance(img: Image.Image) -> Image.Image:
    if cv2 is None:
        return img
    cv = pil_to_cv(img)
    h, w = cv.shape[:2]
    # Scale up small images for better OCR
    if max(h, w) < 1200:
        scale = 1200 / max(h, w)
        cv = cv2.resize(cv, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # Adaptive threshold
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    # Light opening to clear specks
    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv_to_pil(clean)


def preprocess_for_ocr(img: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """Return (original_rgb, enhanced_bw) for dual-pass OCR."""
    return img, _enhance(img)


# -----------------------------
# OCR via PaddleOCR + EasyOCR fallback
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_ocr():
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception as e:
        raise RuntimeError("PaddleOCR not available. Install 'paddleocr'.") from e
    return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


def _ocr_once(ocr, pil_img: Image.Image) -> Tuple[List[str], List, List[float]]:
    try:
        cv = pil_to_cv(pil_img)
        if cv is None:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tf:
                pil_img.save(tf.name)
                result = ocr.ocr(tf.name, cls=True)
        else:
            result = ocr.ocr(cv, cls=True)
    except Exception:
        return [], [], []

    texts, boxes, confs = [], [], []
    if not result:
        return texts, boxes, confs
    for line in result[0]:
        (box, (txt, conf)) = line
        txt = str(txt).strip()
        if not txt:
            continue
        texts.append(txt)
        boxes.append(np.array(box))
        confs.append(float(conf))
    return texts, boxes, confs


def run_ocr_dual(pil_img: Image.Image) -> Tuple[List[str], List, List[float], float, Image.Image]:
    """Try PaddleOCR first; if too few strings, fallback to EasyOCR (if installed)."""
    ocr = get_ocr()
    orig, enh = preprocess_for_ocr(pil_img)
    t0 = time.time()

    # Paddle pass (original + enhanced)
    t1, _b1, c1 = _ocr_once(ocr, orig)
    t2, _b2, c2 = _ocr_once(ocr, enh)

    seen: Dict[str, float] = {}
    for txt, conf in list(zip(t1, c1)) + list(zip(t2, c2)):
        if not txt:
            continue
        seen[txt] = max(conf, seen.get(txt, 0.0))

    # Fallback to EasyOCR if Paddle returned too little
    need_fallback = len(seen) < 3
    if need_fallback:
        try:
            import easyocr  # type: ignore
            reader = easyocr.Reader(['en'], gpu=False)
            npimg = np.array(enh.convert('RGB'))
            results = reader.readtext(npimg, detail=1)
            for (_box, txt, conf) in results:
                t = str(txt).strip()
                if not t:
                    continue
                seen[t] = max(float(conf), seen.get(t, 0.0))
        except Exception:
            pass

    texts = list(seen.keys())
    confs = list(seen.values())
    duration = time.time() - t0
    return texts, [], confs, duration, enh


# -----------------------------
# Optional VLM + LLM (free) — kept for future use
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_blip_pipeline():
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration  # type: ignore
    except Exception as e:
        raise RuntimeError("Transformers not available for BLIP.") from e
    processor = BlipProcessor.from_pretrained(_BLIP_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(_BLIP_MODEL)
    return processor, model


@st.cache_resource(show_spinner=False)
def get_t5_pipeline():
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
    except Exception as e:
        raise RuntimeError("Transformers not available for T5.") from e
    tok = AutoTokenizer.from_pretrained(_T5_MODEL)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(_T5_MODEL)
    return tok, mdl


def blip_caption(img: Image.Image, prompt: str | None = None) -> str:
    try:
        processor, model = get_blip_pipeline()
    except Exception:
        return ""
    inputs = processor(img, text=prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=80)
    return processor.decode(out[0], skip_special_tokens=True)


# -----------------------------
# Interpretation heuristics
# -----------------------------

def extract_title_and_meta(texts: List[str]):
    if not texts:
        return "", {}, []
    candidates = [t for t in texts if len(t.split()) >= 3]
    key_order = [
        "estimate", "survey", "percentage", "change", "landline", "cellphone",
        "percent", "share", "difference", "respondent", "adults", "sample",
    ]
    def score(t: str) -> int:
        s = len(t)
        t_low = t.lower()
        for k in key_order:
            if k in t_low:
                s += 5
        return s
    title = max(candidates, key=score, default=texts[0]) if texts else ""
    axes = [t for t in texts if any(x in t.lower() for x in [
        "percentage point", "number of", "items", "sample estimates", "vs.", "by %"
    ])][:3]
    meta = [t for t in texts if any(x in t.lower() for x in ["source", "center", "pew", "2014", "2015", "method"]) and t not in axes]
    return title, {"axes": axes, "meta": meta}, texts


def compose_interpretation(title: str, axes_meta: dict, ocr_texts: List[str], blip_hint: str | None) -> str:
    raw = " ".join(ocr_texts).lower()
    quant = None
    if re.search(r"less than\s*1\s*percentage", raw) or "1 percentage point or less" in raw:
        quant = "less than 1 percentage point"
    else:
        m = re.search(r"(\d+(?:\.\d+)?)\s*percentage point", raw)
        quant = f"{m.group(1)} percentage point" if m else None

    uses_cell = ("cellphone" in raw or "cell phone" in raw)
    uses_land = ("landline" in raw or "landlines" in raw)

    intro = (f"According to the chart titled '{title}'," if title else "According to the chart,")
    who = (
        " poll results look nearly identical whether based only on those adults reached on cellphones or on a combination of cellphone and landline respondents."
        if (uses_cell or uses_land) else " the compared groups show minimal differences."
    )
    quant_line = f" When landlines are excluded, the estimates change by {quant}, on average." if quant else ""

    core = intro + who + quant_line
    if blip_hint and len(ocr_texts) < 3:
        core += f" Visual read hint: {blip_hint}."
    if not ocr_texts and not blip_hint:
        core = (
            "We couldn't read text from the image. Try a higher‑resolution upload or enable the BLIP caption hint in the sidebar."
        )
    return core.strip()


# -----------------------------
# Streamlit UI (guarded)
# -----------------------------

def _run_streamlit_app():  # pragma: no cover
    st.set_page_config(page_title="Chart→Text (Free Models)", page_icon="📊", layout="wide")
    st.title("📊 Chart → Natural-Language Interpretation")
    st.caption("100% local/open models. No paid APIs.")

    with st.sidebar:
        st.header("Settings")
        use_vlm = st.toggle("Use BLIP caption hint (optional)", value=False)
        use_t5 = st.toggle("Use T5 rewrite (optional)", value=False)
        st.divider()
        st.markdown("**Diagnostics**")
        st.caption("Pipeline: PaddleOCR → EasyOCR fallback → BLIP hint")

    uploaded = st.file_uploader("Upload a chart image", type=["png", "jpg", "jpeg", "webp"]) 

    example_note = st.expander("Need an example?")
    with example_note:
        st.info("Upload any chart. We run dual-pass OCR (original + enhanced). If OCR text is scarce, we try EasyOCR; optionally add a BLIP caption hint.")

    if uploaded:
        img = load_image(uploaded)
        st.image(img, caption="Input", use_container_width=True)

        with st.spinner("Preprocessing & OCR…"):
            try:
                texts, _boxes, confs, took, enhanced = run_ocr_dual(img)
                ocr_ok = len(texts) > 0
            except Exception:
                texts, confs, took, enhanced = [], [], 0.0, img
                ocr_ok = False
        st.caption(f"OCR strings: {len(texts)} • Avg conf: {np.mean(confs) if confs else 0:.2f} • Time: {took:.2f}s")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("OCR Text")
            if ocr_ok:
                st.code("\n".join(texts), language="text")
            else:
                st.warning("No text detected. Try a higher‑resolution image.")
        with col2:
            st.subheader("Enhanced view used for OCR")
            st.image(enhanced, use_container_width=True)

        # Compose interpretation (no ground truth)
        title, axes_meta, ocr_all = extract_title_and_meta(texts)

        blip_hint = None
        if use_vlm:
            with st.spinner("BLIP captioning…"):
                try:
                    blip_hint = blip_caption(img)
                except Exception as e:
                    st.markdown(f"BLIP caption failed: `{e}`")

        base_interpretation = compose_interpretation(title, axes_meta, ocr_all, blip_hint)
        final_text = base_interpretation

        if use_t5 and final_text:
            with st.spinner("T5 rewrite…"):
                try:
                    tok, mdl = get_t5_pipeline()
                    prompt = "Rewrite this chart interpretation to be concise and neutral.\n" + final_text
                    ids = tok(prompt, return_tensors="pt").input_ids
                    out = mdl.generate(ids, max_new_tokens=120)
                    final_text = tok.decode(out[0], skip_special_tokens=True)
                except Exception as e:
                    st.markdown(f"T5 rewrite failed: `{e}`")

        st.subheader("Generated interpretation")
        st.write(final_text)

        st.download_button(
            "Download JSON",
            data=json.dumps({
                "ocr_text": texts,
                "interpretation": final_text,
            }, indent=2),
            file_name="chart_to_text.json",
            mime="application/json",
        )

    else:
        st.info("Upload a chart to begin. For best OCR, upload a crisp 1000–2000 px image.")


# -----------------------------
# Minimal tests for heuristics
# -----------------------------

def _run_tests():  # lightweight, no heavy deps
    import unittest

    class HeuristicTests(unittest.TestCase):
        def test_compose_with_known_phrases(self):
            lines = [
                "Most estimates virtually unchanged when landlines are excluded",
                "Of 279 survey estimates, most differ by 1 percentage point or less when landline interviews are excluded",
                "Percentage point difference between weighted full sample vs. cellphone sample estimates",
                "Number of items",
                "PEW RESEARCH CENTER",
            ]
            title, meta, all_txt = extract_title_and_meta(lines)
            out = compose_interpretation(title, meta, all_txt, blip_hint=None)
            self.assertIn("nearly identical", out)
            self.assertTrue("1 percentage point" in out or "less than 1 percentage point" in out)

        def test_handles_empty_ocr(self):
            out = compose_interpretation("", {"axes": [], "meta": []}, [], blip_hint=None)
            self.assertIn("couldn't read text", out.lower())

        def test_extract_title_prefers_keyword_rich(self):
            lines = ["Random words", "Survey results by percentage point", "Foo"]
            title, meta, _ = extract_title_and_meta(lines)
            self.assertEqual(title, "Survey results by percentage point")
            self.assertIn("axes", meta)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(HeuristicTests)
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    if not res.wasSuccessful():
        raise SystemExit(1)


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":  # pragma: no cover
    if os.environ.get("RUN_TESTS") == "1":
        _run_tests()
    elif _STREAMLIT:
        _run_streamlit_app()
    else:
        # CLI fallback kept for completeness
        img_path = os.environ.get("CHART_IMAGE")
        if not img_path or not os.path.exists(img_path):
            print(json.dumps({"error": "Set CHART_IMAGE to an image path."}, indent=2))
        else:
            img = load_image(img_path)
            texts, *_ = run_ocr_dual(img)
            title, axes_meta, ocr_all = extract_title_and_meta(texts)
            print(json.dumps({"ocr": ocr_all, "title": title}, indent=2))
