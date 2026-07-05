"""Manufacturing Defect Analyzer – Streamlit UI.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# Allow imports from the model/ directory when run via `streamlit run`
ROOT = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT / "model"), str(ROOT / "app")])

from model_loader import DEFAULT_CHECKPOINT, load_model  # noqa: E402
from predict import predict_image  # noqa: E402
from predict_yolo import YOLO_CHECKPOINT, detect_defects, load_yolo  # noqa: E402
from report_generator import generate_report  # noqa: E402
from utils import DEFECT_INFO, LOG_PATH, log_analysis, recommendation  # noqa: E402

st.set_page_config(page_title="Manufacturing Defect Analyzer", page_icon="🔍")

st.title("🔍 Manufacturing Defect Analyzer")
st.caption(
    "AI-powered surface defect inspection for hot-rolled steel "
    "(NEU Surface Defect Database, 6 defect classes)."
)


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_yolo():
    return load_yolo()


if not DEFAULT_CHECKPOINT.exists():
    st.error(
        f"Model checkpoint not found at `{DEFAULT_CHECKPOINT}`.\n\n"
        "Train the model first: `python model/train.py`"
    )
    st.stop()

tab_inspect, tab_dashboard = st.tabs(["🔎 Inspect", "📊 Dashboard"])

with tab_inspect:
    detection_available = YOLO_CHECKPOINT.exists()
    mode = st.radio(
        "Analysis mode",
        ["Classification", "Defect Detection (YOLO)"],
        horizontal=True,
        disabled=not detection_available,
        help=None
        if detection_available
        else "Detection model not trained yet — run `python model/train_yolo.py`",
    )

    uploaded = st.file_uploader(
        "Upload a surface image", type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded is not None:
        image = Image.open(uploaded)
        col_img, col_result = st.columns(2)
        col_img.image(image, caption=uploaded.name, use_container_width=True)

        if st.button("Analyze Image", type="primary"):
            report_args = None

            if mode == "Classification":
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
                    st.progress(
                        p["confidence"], text=f"{title} — {p['confidence']:.1%}"
                    )

                log_analysis(
                    uploaded.name, "classification", result["label"],
                    result["confidence"], verdict,
                )
                report_args = dict(
                    image=image,
                    defect_title=info["title"],
                    description=info["description"],
                    confidence=result["confidence"],
                    verdict=verdict,
                    recommendation=action,
                )
            else:
                with st.spinner("Detecting defects..."):
                    get_yolo()  # warm the cache so repeated runs are fast
                    result = detect_defects(image)

                detections = result["detections"]
                with col_result:
                    st.image(
                        result["annotated_image"],
                        caption="Detected defects",
                        use_container_width=True,
                    )

                if not detections:
                    st.success("No defects detected above the confidence threshold.")
                    log_analysis(uploaded.name, "detection", "none", 0.0, "PASS")
                else:
                    top = max(detections, key=lambda d: d["confidence"])
                    verdict, action = recommendation(top["confidence"])
                    if verdict == "FAIL":
                        st.error(
                            f"Verdict: {verdict} — {len(detections)} defect(s) found"
                        )
                    else:
                        st.warning(
                            f"Verdict: {verdict} — {len(detections)} defect(s) found"
                        )
                    st.markdown(f"**Recommended action:** {action}")

                    st.markdown("**Detections**")
                    for d in detections:
                        info = DEFECT_INFO[d["label"]]
                        st.progress(
                            d["confidence"],
                            text=f"{info['title']} — {d['confidence']:.1%} at {d['box']}",
                        )

                    log_analysis(
                        uploaded.name, "detection", top["label"],
                        top["confidence"], verdict,
                    )
                    labels = sorted({d["label"] for d in detections})
                    report_args = dict(
                        image=result["annotated_image"],
                        defect_title=", ".join(
                            DEFECT_INFO[l]["title"] for l in labels
                        ),
                        description=" ".join(
                            DEFECT_INFO[l]["description"] for l in labels
                        ),
                        confidence=top["confidence"],
                        verdict=verdict,
                        recommendation=action,
                    )

            if report_args:
                report_html = generate_report(filename=uploaded.name, **report_args)
                st.download_button(
                    "Download inspection report (HTML)",
                    data=report_html,
                    file_name=f"inspection_report_{Path(uploaded.name).stem}.html",
                    mime="text/html",
                )

with tab_dashboard:
    if not LOG_PATH.exists():
        st.info("No analyses logged yet. Analyze some images first.")
    else:
        df = pd.read_csv(LOG_PATH)
        df["confidence"] = df["confidence"].astype(float)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total analyses", len(df))
        c2.metric("Avg. confidence", f"{df['confidence'].mean():.1%}")
        c3.metric("FAIL verdicts", int((df["verdict"] == "FAIL").sum()))
        c4.metric("Defect types seen", df.loc[df["label"] != "none", "label"].nunique())

        st.markdown("**Defect type distribution**")
        counts = df.loc[df["label"] != "none", "label"].value_counts()
        st.bar_chart(counts)

        st.markdown("**Verdict breakdown**")
        st.bar_chart(df["verdict"].value_counts())

        st.markdown("**Recent analyses**")
        st.dataframe(
            df.sort_values("timestamp", ascending=False).head(20),
            use_container_width=True,
            hide_index=True,
        )
