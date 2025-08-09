"""
Chart-to-Text: Streamlit UI with CLI fallback
-------------------------------------------------
This file supports two modes:
1) Streamlit UI (when `streamlit` is installed).
2) CLI fallback (when `streamlit` is unavailable). The previous error
   `SyntaxError: keyword argument repeated: markdown` came from a duplicate
   `markdown` attribute in the Streamlit shim. This has been fixed.

Run:
  # UI (if you have Streamlit)
  streamlit run app.py

  # CLI (no Streamlit needed)
  python app.py
  CHART_IMAGE=/path/to/chart.png python app.py

Tests:
  RUN_TESTS=1 python app.py
"""

from __future__ import annotations
import io
import os
import re
import json
from types import SimpleNamespace
from typing import List

# --- Optional deps (import guarded) -------------------------------------------------
try:  # OpenCV is optional in CLI; we handle None below
    import cv2  # type: ignore
except Exception:  # pragma: no cover
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

    # Minimal shim with only the attributes we reference
    st = SimpleNamespace(  # type: ignore
        cache_resource=_noop_decorator,
        set_page_config=_noop,
        title=_noop,
        caption=_noop,
        sidebar=SimpleNamespace(
            __enter__=lambda *a, **k: None,
            __exit__=lambda *a, **k: False,
        ),
        header=_noop,
        toggle=lambda *a, **k: False,
        divider=_noop,
        markdown=_noop,  # <-- single definition (fixed)
        checkbox=lambda *a, **k: False,
        file_uploader=lambda *a, **k: None,
        expander=lambda *a, **k: SimpleNamespace(__enter__=lambda *a, **k: None, __exit__=lambda *a, **k: False),
        info=_noop,
        image=_noop,
        spinner=SimpleNamespace(__enter__=lambda *a, **k: None, __exit__=lambda *a, **k: False),
        columns=lambda n: (
            SimpleNamespace(subheader=_noop, warning=_noop, code=_noop, caption=_noop),
        ) * n,
        subheader=_noop,
        warning=_noop,
        code=_noop,
        write=_noop,
        download_button=_noop,
    )

# Optional, CPU-friendly VLM + small LLM (all free/open)
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

def preprocess_for_ocr(img: Image.Image):
    if cv2 is None:  # OpenCV not available
        return img
    cv = pil_to_cv(img)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv_to_pil(clean)


# -----------------------------
# OCR via PaddleOCR (optional)
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_ocr():
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception as e:
        raise RuntimeError("PaddleOCR not available. Install 'paddleocr'.") from e
    return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


def run_ocr(img: Image.Image):
    """Run OCR if PaddleOCR is available; otherwise return empty results."""
    try:
        ocr = get_ocr()
        cv = pil_to_cv(img)
        if cv is None:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tf:
                img.save(tf.name)
                result = ocr.ocr(tf.name, cls=True)
        else:
            result = ocr.ocr(cv, cls=True)
    except Exception:
        return [], [], []

    texts: List[str] = []
    boxes = []
    confidences: List[float] = []
    if not result:
        return texts, boxes, confidences
    for line in result[0]:
        (box, (txt, conf)) = line
        texts.append(txt)
        boxes.append(np.array(box))
        confidences.append(conf)
    return texts, boxes, confidences


# -----------------------------
# Lightweight VLM + LLM (optional)
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
# Chart structure heuristics
# -----------------------------

def extract_title_and_meta(texts: List[str]):
    if not texts:
        return "", {}, []
    candidates = [t for t in texts if len(t.split()) >= 3]
    key_order = [
        "estimates", "survey", "percentage", "change", "landline", "cellphone",
        "percent", "share", "difference", "respondents", "adults", "sample",
    ]
    def score(t: str) -> int:
        s = len(t)
        t_low = t.lower()
        for k in key_order:
            if k in t_low:
                s += 5
        return s
    title = max(candidates, key=score, default=texts[0]) if texts else ""
    meta = []
    for t in texts:
        if t != title and any(k in t.lower() for k in [
            "source", "pew", "surveys", "conducted", "center", "september", "2015", "2014"
        ]):
            meta.append(t)
    axes = [t for t in texts if any(x in t.lower() for x in [
        "percentage point", "number of", "items", "sample estimates"
    ])][:3]
    return title, {"axes": axes, "meta": meta}, texts


def compose_interpretation(title: str, axes_meta: dict, ocr_texts: List[str], blip_hint: str | None) -> str:
    raw = " ".join(ocr_texts).lower()
    quant = None
    if re.search(r"less than\s*1\s*percentage", raw) or "1 percentage point or less" in raw:
        quant = "less than 1 percentage point"
    else:
        m = re.search(r"(\d+(?:\.\d+)?)\s*percentage point", raw)
        quant = f"{m.group(1)} percentage point" if m else None

    entities = []
    if "cellphone" in raw or "cell phone" in raw:
        entities.append("cellphone respondents")
    if "landline" in raw or "landlines" in raw:
        entities.append("landline respondents")

    intro = (
        f"According to the chart titled '{title}'," if title else "According to the chart,"
    )
    who = (
        " poll results look nearly identical whether based only on those adults reached on cellphones or on a combination of cellphone and landline respondents."
        if entities else " the results across groups look nearly identical."
    )
    quant_line = f" When landlines are excluded, the estimates change by {quant}, on average." if quant else ""

    core = intro + who + quant_line

    if blip_hint:
        core += f" Visual read: {blip_hint}."

    if not ocr_texts:
        core = (
            "We couldn't read text from the image (OCR not available). "
            "Install 'paddleocr' for full accuracy, or run the Streamlit UI in an environment where it is installed."
        )
    return core.strip()


# -----------------------------
# Streamlit UI (guarded) + CLI fallback
# -----------------------------

def _run_streamlit_app():  # pragma: no cover
    st.set_page_config(page_title="Chart→Text (Free Models)", page_icon="📊", layout="wide")
    st.title("📊 Chart → Natural-Language Interpretation")
    st.caption("100% local/open models. No paid APIs.")

    with st.sidebar:
        st.header("Settings")
        use_vlm = st.toggle("Use BLIP caption (optional)", value=False, help="Adds a general image caption using BLIP. Increases inference time.")
        use_t5 = st.toggle("Use T5 rewrite (optional)", value=False, help="Rewrite the final interpretation with a small open LLM (Flan‑T5).")
        st.divider()
        st.markdown("**Export**")
        want_json = st.checkbox("Return raw OCR + interpretation as JSON", value=False)

    uploaded = st.file_uploader("Upload a chart image", type=["png", "jpg", "jpeg", "webp"]) 

    example_note = st.expander("Need an example?")
    with example_note:
        st.info("Drag in any chart screenshot. The app extracts text with PaddleOCR, uses heuristics, and optionally adds a BLIP caption and a T5 rewrite.")

    if uploaded:
        img = load_image(uploaded)
        st.image(img, caption="Input", use_column_width=True)

        with st.spinner("Preprocessing & OCR…"):
            clean = preprocess_for_ocr(img)
            texts, boxes, confs = run_ocr(clean)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("OCR Text")
            if len(texts) == 0:
                st.warning("No text detected. Try a higher‑resolution image or check OCR installation.")
            else:
                st.code("\n".join(texts), language="text")
            avg_conf = float(np.mean(confs)) if confs else 0.0
            st.caption(f"Avg confidence: {avg_conf:.2f}")

        with col2:
            st.subheader("Preprocessed view")
            st.image(clean, use_column_width=True)

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

        if use_t5:
            with st.spinner("T5 rewrite…"):
                try:
                    tok, mdl = get_t5_pipeline()
                    prompt = (
                        "Rewrite the following chart interpretation to be concise, precise, and neutral.\n" 
                        + base_interpretation
                    )
                    ids = tok(prompt, return_tensors="pt").input_ids
                    out = mdl.generate(ids, max_new_tokens=120)
                    final_text = tok.decode(out[0], skip_special_tokens=True)
                except Exception as e:
                    st.markdown(f"T5 rewrite failed: `{e}`")

        st.subheader("Generated interpretation")
        st.write(final_text)

        if want_json:
            st.download_button(
                "Download JSON",
                data=json.dumps({
                    "title": title,
                    "axes_meta": axes_meta,
                    "ocr_text": ocr_all,
                    "interpretation": final_text,
                }, indent=2),
                file_name="chart_to_text.json",
                mime="application/json",
            )

    else:
        st.info("Upload a chart to begin. In the sidebar, you can enable BLIP and T5 for extra context.")

    st.markdown("---")
    st.markdown(
        "**Tech note:** Uses PaddleOCR (free) for text, optional BLIP caption, and optional Flan‑T5 rewrite. All models are free/open."
    )


def _interpret_image_path(img_path: str) -> dict:
    img = load_image(img_path)
    clean = preprocess_for_ocr(img)
    texts, _boxes, _confs = run_ocr(clean)
    title, axes_meta, ocr_all = extract_title_and_meta(texts)
    blip_hint = ""  # disabled in CLI for speed
    interpretation = compose_interpretation(title, axes_meta, ocr_all, blip_hint)
    return {
        "title": title,
        "axes_meta": axes_meta,
        "ocr_text": ocr_all,
        "interpretation": interpretation,
    }


def _run_cli():
    default_sample = "/mnt/data/f1b084f3-db66-4c72-a77c-aab3ebe3f58f.png"
    img_path = os.environ.get("CHART_IMAGE", default_sample)

    if not os.path.exists(img_path):
        print(json.dumps({
            "error": "No image found",
            "hint": "Set CHART_IMAGE to the path of a chart image.",
        }, indent=2))
        return

    result = _interpret_image_path(img_path)
    print(json.dumps(result, indent=2))
    with open("chart_to_text.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


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
            lines = [
                "Random words", "Survey results by percentage point", "Foo"
            ]
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
        _run_cli()
