"""Streamlit web app:  streamlit run app.py"""
import html
import re

import streamlit as st

from fakenews.config import CLASS_NAMES
from fakenews.explain import METHODS, explain, plot_contributions
from fakenews.registry import available_models, load_model

st.set_page_config(page_title="Explainable Fake-News Detection", page_icon="📰", layout="centered")
st.title("📰 Explainable Fake-News Detection")
st.caption("Truth with Reasons - classify a news text and see *why* the model decided.")

models = available_models()
if not models:
    st.error("No trained model found. Run this in a terminal first:\n\n"
             "`python train.py --dataset sample --models nb lr`")
    st.stop()


@st.cache_resource(show_spinner="Loading model...")
def get_model(name: str):
    return load_model(name)


with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", models)
    method = st.selectbox("Explainer", METHODS, index=METHODS.index("occlusion"),
                          help="LIME / SHAP need the packages installed; otherwise occlusion is used.")
    top_k = st.slider("Words to show", 3, 15, 8)

EXAMPLES = {
    "Suspicious": "Scientists discover cure for cancer in just 5 days!",
    "Plausible": "New study shows promising results for cancer treatment in early clinical trials.",
    "Policy news": "The government has announced a new tax cut for all citizens.",
}
choice = st.radio("Try an example or write your own:", ["Custom", *EXAMPLES], horizontal=True)
default = "" if choice == "Custom" else EXAMPLES[choice]
text = st.text_area("News text", value=default, height=160, key=f"text-{choice}")

if st.button("Analyze", type="primary") and text.strip():
    model = get_model(model_name)
    proba = model.predict_proba([text])[0]
    label = CLASS_NAMES[int(proba.argmax())]

    (st.error if label == "FAKE" else st.success)(f"Prediction: **{label}**  (confidence {proba.max():.1%})")
    c1, c2 = st.columns(2)
    c1.metric("P(REAL)", f"{proba[0]:.1%}")
    c2.metric("P(FAKE)", f"{proba[1]:.1%}")

    with st.spinner(f"Explaining with {method}..."):
        pairs = explain(method, model.predict_proba, text, top_k=top_k)

    if not pairs:
        st.info("No individual word had a measurable influence on this prediction.")
    else:
        st.subheader("Why?")
        st.pyplot(plot_contributions(pairs))

        weights = {w.lower(): v for w, v in pairs}
        top = max(abs(v) for v in weights.values())

        def mark(match):
            word = match.group(0)
            v = weights.get(word.lower())
            if v is None:
                return html.escape(word)
            alpha = 0.2 + 0.6 * abs(v) / top
            color = f"rgba(214,39,40,{alpha:.2f})" if v > 0 else f"rgba(44,160,44,{alpha:.2f})"
            return f'<span style="background:{color};padding:1px 3px;border-radius:3px">{html.escape(word)}</span>'

        st.markdown("**Highlighted text** (red = toward FAKE, green = toward REAL)")
        st.markdown(re.sub(r"[\w']+", mark, text).replace("\n", "<br>"), unsafe_allow_html=True)
