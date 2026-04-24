"""
rag/rag_pipeline.py
RAG Pipeline for Fleet Management Insight Assistant
- Converts dataset rows to text documents
- Stores embeddings in FAISS
- Retrieves relevant rows for a query
- Generates answers using Claude API (with rule-based fallback)
"""

import os
import json
import numpy as np
import pandas as pd
import re

# ─────────────────────────────────────────────
# DOCUMENT BUILDER
# ─────────────────────────────────────────────

def build_documents(df: pd.DataFrame) -> list[dict]:
    """Convert each DataFrame row into a text document for RAG."""
    docs = []
    for _, row in df.iterrows():
        parts = []
        for col, val in row.items():
            if not col.endswith("_norm") and not col.startswith("anomaly"):
                parts.append(f"{col.replace('_', ' ')}: {val}")
        text = " | ".join(str(p) for p in parts)
        docs.append({"text": text, "metadata": row.to_dict()})
    return docs


# ─────────────────────────────────────────────
# VECTOR STORE (FAISS)
# ─────────────────────────────────────────────

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FleetVectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.vectors = None
        self.documents = []

    def build(self, documents: list[dict]):
        """Convert documents into TF-IDF vectors."""
        self.documents = documents
        texts = [d["text"] for d in documents]
        self.vectors = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top-k most relevant documents."""
        if self.vectors is None:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.vectors).flatten()

        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc["score"] = float(similarities[idx])
            results.append(doc)

        return results


# ─────────────────────────────────────────────
# RULE-BASED QUERY ANSWERING (Fallback)
# ─────────────────────────────────────────────

def rule_based_answer(query: str, df: pd.DataFrame) -> str:
    """Simple keyword-based query answering on the fleet dataframe."""
    q = query.lower()

    # Aggregate by vehicle for summary questions
    if "vehicle_id" not in df.columns:
        return "Dataset does not contain vehicle_id column."

    vdf = df.groupby("vehicle_id").mean(numeric_only=True).reset_index()

    # --- Highest / Most ---
    if re.search(r"highest|most|maximum|max|worst", q):
        if "fuel" in q and "consumption" in q:
            row = vdf.loc[vdf["fuel_consumption"].idxmax()]
            return f"🚛 Vehicle **{row['vehicle_id']}** has the highest average fuel consumption: **{row['fuel_consumption']:.2f} L**."
        if "maintenance" in q and "cost" in q:
            row = vdf.loc[vdf["maintenance_cost"].idxmax()]
            return f"🔧 Vehicle **{row['vehicle_id']}** has the highest average maintenance cost: **₹{row['maintenance_cost']:.2f}**."
        if "distance" in q:
            row = vdf.loc[vdf["distance_travelled"].idxmax()]
            return f"📍 Vehicle **{row['vehicle_id']}** has travelled the most on average: **{row['distance_travelled']:.2f} km**."
        if "driver" in q or "behavior" in q or "score" in q:
            row = vdf.loc[vdf["driver_behavior_score"].idxmax()]
            return f"🏅 Vehicle **{row['vehicle_id']}** has the highest driver behavior score: **{row['driver_behavior_score']:.1f}**."

    # --- Lowest / Least / Best ---
    if re.search(r"lowest|least|minimum|min|best", q):
        if "fuel" in q and "consumption" in q:
            row = vdf.loc[vdf["fuel_consumption"].idxmin()]
            return f"✅ Vehicle **{row['vehicle_id']}** has the lowest average fuel consumption: **{row['fuel_consumption']:.2f} L**."
        if "maintenance" in q and "cost" in q:
            row = vdf.loc[vdf["maintenance_cost"].idxmin()]
            return f"✅ Vehicle **{row['vehicle_id']}** has the lowest average maintenance cost: **₹{row['maintenance_cost']:.2f}**."
        if "driver" in q or "behavior" in q or "score" in q:
            row = vdf.loc[vdf["driver_behavior_score"].idxmin()]
            return f"⚠️ Vehicle **{row['vehicle_id']}** has the lowest driver behavior score: **{row['driver_behavior_score']:.1f}**."

    # --- High maintenance ---
    if re.search(r"high maintenance|expensive maintenance|costly", q):
        threshold = vdf["maintenance_cost"].quantile(0.75)
        high = vdf[vdf["maintenance_cost"] >= threshold]["vehicle_id"].tolist()
        return f"🔧 Vehicles with **high maintenance cost** (top 25%): **{', '.join(high)}**."

    # --- Fuel efficiency ---
    if "fuel efficiency" in q or "efficient" in q:
        if "fuel_efficiency" in vdf.columns:
            row = vdf.loc[vdf["fuel_efficiency"].idxmax()]
            return f"⚡ Vehicle **{row['vehicle_id']}** is most fuel-efficient with **{row['fuel_efficiency']:.2f} km/L**."

    # --- Average ---
    if "average" in q or "mean" in q:
        if "fuel" in q:
            return f"📊 Average fuel consumption across all vehicles: **{df['fuel_consumption'].mean():.2f} L**."
        if "maintenance" in q:
            return f"📊 Average maintenance cost across all vehicles: **₹{df['maintenance_cost'].mean():.2f}**."
        if "distance" in q:
            return f"📊 Average distance travelled: **{df['distance_travelled'].mean():.2f} km**."
        if "driver" in q or "score" in q:
            return f"📊 Average driver behavior score: **{df['driver_behavior_score'].mean():.1f}**."

    # --- Count vehicles ---
    if "how many vehicle" in q or "total vehicle" in q or "number of vehicle" in q:
        return f"🚗 Total unique vehicles in the fleet: **{df['vehicle_id'].nunique()}**."

    # --- List vehicles ---
    if "list" in q and "vehicle" in q:
        vehicles = df["vehicle_id"].unique().tolist()
        return f"🚗 Vehicles in fleet: **{', '.join(vehicles)}**."

    # --- Specific vehicle ---
    match = re.search(r"v\d{3}", q)
    if match:
        vid = match.group().upper()
        vdata = df[df["vehicle_id"] == vid]
        if not vdata.empty:
            avg = vdata.mean(numeric_only=True)
            return (
                f"🔍 **{vid}** Summary:\n"
                f"- Avg Distance: **{avg.get('distance_travelled', 0):.1f} km**\n"
                f"- Avg Fuel Consumption: **{avg.get('fuel_consumption', 0):.2f} L**\n"
                f"- Avg Maintenance Cost: **₹{avg.get('maintenance_cost', 0):.2f}**\n"
                f"- Avg Driver Score: **{avg.get('driver_behavior_score', 0):.1f}**\n"
                f"- Avg Fuel Efficiency: **{avg.get('fuel_efficiency', 0):.2f} km/L**"
            )
        return f"❌ Vehicle **{vid}** not found in dataset."

    return (
        "🤔 I couldn't find a direct answer to your query. Try asking:\n"
        "- *Which vehicle has highest fuel consumption?*\n"
        "- *Show vehicles with high maintenance cost*\n"
        "- *What is the average distance travelled?*\n"
        "- *Details of V001*"
    )


# ─────────────────────────────────────────────
# LLM-ENHANCED ANSWERING (Gemini API)
# ─────────────────────────────────────────────

def generate_rag_answer(
    query: str,
    df: pd.DataFrame,
    vector_store: FleetVectorStore,
    api_key: str = None,
    use_llm: bool = False
) -> str:
    """
    Full RAG answer:
    1. Retrieve relevant rows via FAISS
    2. If LLM API key available → call Gemini
    3. Else → rule-based fallback
    """
    # --- Retrieve context ---
    retrieved = vector_store.search(query, top_k=5)
    context_rows = [r["text"] for r in retrieved]
    context_str = "\n".join(context_rows[:5])

    if use_llm and api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name='gemini-2.5-flash')
            prompt = (
                f"You are a fleet management assistant. Use ONLY the context below to answer the question.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {query}\n\n"
                f"Answer concisely and clearly:"
            )
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"[LLM Error: {e}]\n\n" + rule_based_answer(query, df)

    # --- Rule-based fallback ---
    return rule_based_answer(query, df)
