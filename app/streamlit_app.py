"""Manufacturing Defect Analyzer – Streamlit UI.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# Allow imports from the model/ directory when run via `streamlit run`
ROOT = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT / "model"), str(ROOT / "app")])

from model_loader import DEFAULT_CHECKPOINT, load_model  # noqa: E402
from predict import predict_image  # noqa: E402
from report_generator import generate_report  # noqa: E402
from utils import DEFECT_INFO, recommendation  # noqa: E402

st.set_page_config(page_title="Manufacturing Defect Analyzer", page_icon="🔍")

st.title("🔍 Manufacturing Defect Analyzer")
st.caption(
    "AI-powered surface defect classification for hot-rolled steel "
    "(NEU Surface Defect Database, 6 defect classes)."
)


@st.cache_resource
def get_model():
    return load_model()


if not DEFAULT_CHECKPOINT.exists():
    st.error(
        f"Model checkpoint not found at `{DEFAULT_CHECKPOINT}`.\n\n"
        "Train the model first: `python model/train.py`"
    )
    st.stop()

uploaded = st.file_uploader(
    "Upload a surface image", type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded is not None:
    image = Image.open(uploaded)
    col_img, col_result = st.columns(2)
    col_img.image(image, caption=uploaded.name, use_container_width=True)

    if st.button("Analyze Image", type="primary"):
        with st.spinner("Analyzing..."):
            result = predict_image(image, model=get_model())

        info = DEFECT_INFO[result["label"]]
        verdict, action = recommendation(result["confidence"])

        with col_result:
            st.subheader(info["title"])
            st.metric("Confidence", f"{result['confidence']:.1%}")
            if verdict == "FAIL":
                st.error(f"Verdict: {verdict}")
            else:
                st.warning(f"Verdict: {verdict}")

        st.markdown(f"**What is this defect?** {info['description']}")
        st.markdown(f"**Recommended action:** {action}")

        st.markdown("**Top predictions**")
        for p in result["top_predictions"]:
            title = DEFECT_INFO[p["label"]]["title"]
            st.progress(p["confidence"], text=f"{title} — {p['confidence']:.1%}")

        report_html = generate_report(
            image=image,
            filename=uploaded.name,
            defect_title=info["title"],
            description=info["description"],
            confidence=result["confidence"],
            verdict=verdict,
            recommendation=action,
        )
        st.download_button(
            "Download inspection report (HTML)",
            data=report_html,
            file_name=f"inspection_report_{Path(uploaded.name).stem}.html",
            mime="text/html",
        )
