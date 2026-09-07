"""
Steg 4: Streamlit - samma artefakt från samma register, annat gränssnitt.
Kör:  uv run streamlit run src/iris_mlflow/ui.py
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import streamlit as st
from mlflow.models import get_model_info

from iris_mlflow.config import ALIAS, MODEL_NAME, MODEL_URI, TRACKING_URI


@st.cache_resource  # laddas en gång, inte vid varje klick
def load_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    info = get_model_info(MODEL_URI)
    model = mlflow.sklearn.load_model(MODEL_URI)
    feature_names = [col["name"] for col in info.signature.inputs.to_dict()]
    version = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS).version
    return model, feature_names, info.metadata["target_names"], version


model, feature_names, target_names, version = load_model()

st.title("Iris-klassificerare")
st.caption(f"{MODEL_NAME} version {version} (alias '{ALIAS}')")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1, 0.1)
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5, 0.1)
with col2:
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4, 0.1)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2, 0.1)

values = {
    "sepal length (cm)": sepal_length,
    "sepal width (cm)": sepal_width,
    "petal length (cm)": petal_length,
    "petal width (cm)": petal_width,
}
X = pd.DataFrame([[values[n] for n in feature_names]], columns=feature_names)
class_index = int(model.predict(X)[0])
probas = model.predict_proba(X)[0]

st.subheader(f"Prediktion: **{target_names[class_index]}**")
st.bar_chart({name: float(p) for name, p in zip(target_names, probas)})
