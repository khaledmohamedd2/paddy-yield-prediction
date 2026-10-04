# 🌾 Paddy Yield Prediction

A machine learning web application that predicts paddy (rice) yield in Kg based on farm characteristics, agricultural inputs, and weather conditions.

## Overview

Paddy yield depends on many factors: field size, soil type, rice variety, fertilizer and pesticide use, rainfall, irrigation, temperature, wind, and humidity. This project uses a **Random Forest Regressor** trained on a real paddy farming dataset to estimate the expected yield, and exposes it through an interactive **Streamlit** dashboard.

## Features

- Clean, modern agricultural dashboard
- 44 input features organized into logical sections:
  - Farm Information
  - Soil & Seed Information
  - Fertilizer & Pest Management
  - Rainfall & Irrigation
  - Temperature
  - Wind & Humidity
  - Other Information
- Helpful tooltips on inputs
- Large result card showing the predicted yield in Kg

## Tech Stack

- **Python**
- **Pandas / NumPy** for data handling
- **scikit-learn** for preprocessing and the Random Forest model
- **Streamlit** for the web interface
- **Joblib** for model serialization

## Project Structure

```
├── app.py                    # Streamlit application
├── random_forest_model.pkl   # Trained Random Forest model
├── preprocessor.pkl          # Fitted preprocessing pipeline
├── paddydataset.csv          # Dataset
├── requirements.txt          # Dependencies
└── README.md
```

## How to Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/paddy-yield-prediction.git
cd paddy-yield-prediction
pip install -r requirements.txt
python -m streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## How It Works

1. The user enters farm, input, and weather values in the dashboard.
2. The values pass through the saved preprocessor (encoding and scaling).
3. The Random Forest model predicts the yield.
4. The result is displayed in Kg.

## Dataset

The dataset contains paddy farming records with details on field size, soil, variety, fertilizer and pesticide applications, rainfall, irrigation, temperature, wind, humidity, and the final yield.
