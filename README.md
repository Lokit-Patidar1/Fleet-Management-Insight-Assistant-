# 🚛 Fleet Management Insight Assistant

A college-level major project demonstrating AI/ML in fleet operations analytics.

---

## 📁 Project Structure

```
fleet_management_insight_assistant/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── data/
│   └── fleet_data.csv        # Sample fleet dataset
├── models/
│   ├── __init__.py
│   └── ml_models.py          # Regression + Anomaly Detection
├── rag/
│   ├── __init__.py
│   └── rag_pipeline.py       # FAISS vector store + RAG pipeline
└── utils/
    ├── __init__.py
    ├── data_processor.py     # Preprocessing & feature engineering
    └── charts.py             # Plotly visualizations
```

---

## ⚡ Quick Setup

### 1. Clone / Download the project folder

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Configure environment (optional for LLM)
Create a `.env` file in the project root and add your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🎯 Features

| Feature | Details |
|---|---|
| 📊 Dashboard | KPI cards, fuel consumption, maintenance cost trends, driver behavior gauge |
| 🤖 ML — Regression | Random Forest predicts maintenance cost; shows MAE, RMSE, R², feature importance |
| 🤖 ML — Anomaly | Isolation Forest detects unusual vehicle records |
| 💬 RAG Query | Natural language queries answered using FAISS retrieval + rule-based engine |
| 🔑 LLM (Optional) | Set GEMINI_API_KEY in .env to enable Gemini-powered answers |
| 📂 CSV Upload | Upload your own fleet CSV from the sidebar |

---

## 📊 Dataset Fields

| Column | Description |
|---|---|
| `vehicle_id` | Unique vehicle identifier |
| `date` | Record date |
| `distance_travelled` | Distance in km |
| `fuel_consumption` | Fuel used in litres |
| `maintenance_cost` | Maintenance cost in ₹ |
| `driver_behavior_score` | Score 0–100 |
| `vehicle_type` | Car / Van / Truck |
| `driver_id` | Driver identifier |

---

## 🧠 ML Architecture

```
Raw CSV → Preprocessing → Feature Engineering
                                 ↓
                    ┌────────────┴────────────┐
                    │                         │
              Regression                 Anomaly Detection
         (Random Forest)              (Isolation Forest)
         Predict maintenance          Detect unusual records
               cost
                    │                         │
                    └────────────┬────────────┘
                                 ↓
                          Streamlit UI
                                 ↓
                    ┌────────────┴────────────┐
                    │                         │
              FAISS Vector Store         Rule-based NLP
              (sentence-transformers)    (keyword patterns)
                    └────────────┬────────────┘
                                 ↓
                        Natural Language Answers
```

---

## ☁️ Deploy to Streamlit Cloud

1. Push the folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set `app.py` as entrypoint
4. Deploy — it's free!

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML**: scikit-learn (RandomForest, IsolationForest)
- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **Charts**: Plotly Express
- **LLM (optional)**: Google Gemini API
