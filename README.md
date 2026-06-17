# Crop Yield Predictor and Interpreter

A machine learning web application that predicts crop yields and provides explainable AI insights using SHAP, integrated with real-time NASA POWER meteorological data.

Live Repository: https://github.com/huzaib123/Crop-Yield-Predictor-

---

## Key Features

*   **Explainable AI**: Implements SHAP (SHapley Additive exPlanations) to decompose model predictions and visualize feature impact (both positive and negative drivers) in real time.
*   **Application Security**: Features a custom security layer providing session-based rate-limiting, input sanitization, path-traversal prevention, and error-scrubbing.
*   **Meteorological Integration**: Automatically queries historical and daily climate observations (temperature, precipitation, solar radiation) for specific coordinates via the NASA POWER API.
*   **Clean Architecture**: Structured with a clear separation between the modular training pipelines, presentation layer, and serialized inference objects.

---

## Tech Stack

*   **Frontend**: Streamlit with custom CSS layout and responsive design.
*   **Machine Learning**: Scikit-Learn, Joblib, and SHAP for model interpretability.
*   **Data & APIs**: Pandas, NumPy, and Requests to query the NASA POWER API.
*   **Security**: Regular expression sanitization and session-state rate limiting.

---

## Directory Structure

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

## Quick Start

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