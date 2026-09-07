"""
Steg 4: Streamlit - samma modell, annat gränssnitt.

Kör:  uv run streamlit run src/iris_service/ui.py

Jämför med api.py: exakt samma joblib-fil, samma pipeline, ingen
webbserver-kod. Streamlit passar för demo/intern verktyg; FastAPI passar
när andra system ska anropa modellen. Gruppens val i projektet.
"""

import json

import joblib
import numpy as np
import streamlit as st

from iris_service.config import META_PATH, MODEL_PATH


@st.cache_resource  # laddas en gång, inte vid varje klick
def load_model():
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text())


model, meta = load_model()

st.title("Iris-klassificerare")
st.caption(f"Modell tränad med scikit-learn {meta['sklearn_version']} "
           f"- testaccuracy {meta['test_accuracy']:.1%}")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1, 0.1)
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5, 0.1)
with col2:
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4, 0.1)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2, 0.1)

X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
class_index = int(model.predict(X)[0])
probas = model.predict_proba(X)[0]

st.subheader(f"Prediktion: **{meta['target_names'][class_index]}**")
st.bar_chart(
    {name: float(p) for name, p in zip(meta["target_names"], probas)}
)
