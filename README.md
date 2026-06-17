# 🌾 Crop Yield Predictor & Interpreter

A production-ready, security-hardened Machine Learning web application that predicts crop yields and provides explainable AI (XAI) insights using SHAP, integrated with real-time NASA POWER meteorological data.

🔗 **Live Code Repository**: [GitHub](https://github.com/huzaib123/Crop-Yield-Predictor-)

---

## 🚀 Key Highlights

*   **Explainable AI (XAI)**: Implements SHAP (SHapley Additive exPlanations) to decompose model predictions and visualize feature impact (positive/negative drivers) in real time.
*   **Production-Grade Security**: Features an application-level firewall (`security.py`) providing session-based rate-limiting, strict regex input validation, path-traversal prevention, and error-scrubbing to prevent source code leaks.
*   **NASA Meteorological Integration**: Directly pulls historical and daily climate observations (temperature, precipitation, solar radiation) for precise coordinates via the NASA POWER API.
*   **Clean Architecture**: Segregated into modular training pipelines (`src/`), a secure presentation layer (`app/`), and serialized inference objects (`models/`).

---

## 🛠️ Architecture & Tech Stack

*   **Frontend**: Streamlit, Custom Vanilla CSS (Premium Glassmorphism & Micro-animations)
*   **Machine Learning**: Scikit-Learn, Joblib, SHAP (Interpretability)
*   **Data & APIs**: Pandas, NumPy, OkHttp/Requests, NASA POWER API
*   **Security**: Regular Expression Sanitization, Session-State Rate Limiting

---

## 📦 Directory Structure

```text
├── app/
│   ├── app.py           # Main application entry point & UI
│   ├── security.py      # Security firewall (sanitization & rate-limiting)
│   ├── map_view.py      # Spatial visualization components
│   └── shap_view.py     # SHAP contribution charts
├── src/
│   ├── train.py         # Pipeline training script
│   └── preprocess.py    # Data cleansing and transformation
├── models/              # Serialized pipeline binaries (.pkl)
└── data/                # Data storage directories
```

---

## ⚡ Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/huzaib123/Crop-Yield-Predictor-.git
   cd Crop-Yield-Predictor-
   ```

2. **Set up virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app/app.py
   ```