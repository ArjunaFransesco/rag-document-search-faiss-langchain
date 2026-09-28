import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Enterprise RAG Document Knowledge Assistant & Semantic Search",
    page_icon="🤖",
    layout="wide"
)

st.title("🎯 Enterprise RAG Document Knowledge Assistant & Semantic Search")
st.markdown("**Domain**: `Deep Learning / Generative AI & NLP` | **Tech Stack**: `LangChain, FAISS, PyTorch Embeddings, Streamlit`")
st.markdown("**Author**: [Arjuna Fransesco](https://github.com/ArjunaFransesco) | **GitHub**: [Portfolio Repositories](https://github.com/ArjunaFransesco?tab=repositories)")
st.markdown("---")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("⚙️ Domain Input Telemetry")
    query_token_length = st.slider("Query Token Length", int(3), int(64), int(18))
    dense_cosine_similarity = st.slider("Dense Cosine Similarity", float(0.2), float(0.99), float(0.78))
    bm25_lexical_score = st.slider("Bm25 Lexical Score", float(1.0), float(35.0), float(14.5))
    document_chunk_tokens = st.slider("Document Chunk Tokens", int(64), int(512), int(256))
    cross_encoder_rerank_score = st.slider("Cross Encoder Rerank Score", float(0.05), float(0.99), float(0.82))
    reciprocal_rank_fusion_score = st.slider("Reciprocal Rank Fusion Score", float(0.005), float(0.08), float(0.032))

with col2:
    st.subheader("🔮 Predictive Model Inference")
    model_path = os.path.join(os.path.dirname(__file__), "models/model_pipeline.joblib")
    scaler_path = os.path.join(os.path.dirname(__file__), "models/scaler.joblib")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        input_df = pd.DataFrame([{"query_token_length": query_token_length, "dense_cosine_similarity": dense_cosine_similarity, "bm25_lexical_score": bm25_lexical_score, "document_chunk_tokens": document_chunk_tokens, "cross_encoder_rerank_score": cross_encoder_rerank_score, "reciprocal_rank_fusion_score": reciprocal_rank_fusion_score}])
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        
        st.markdown("#### Real-Time Prediction Output")
        st.info(f"Predicted `is_relevant_passage`: **{pred}**")
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)[0]
            st.progress(float(probs[1]) if len(probs) > 1 else float(probs[0]))
            st.caption(f"Confidence Probability Score: **{np.max(probs):.2%}**")
    else:
        st.warning("Model or Scaler artifact not found in models/ directory.")

st.markdown("---")
st.markdown("### 📊 Benchmark Metrics")
metrics_path = os.path.join(os.path.dirname(__file__), "reports/metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    st.json(metrics_data)
