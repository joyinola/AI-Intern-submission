import joblib
import numpy as np
import pandas as pd

model = joblib.load("models/risk_model.joblib")

def risk_band(prob):
    if prob < 0.25:
        return "Green"
    elif prob < 0.6:
        return "Yellow"
    return "Red"

def recommendations(band):
    if band == "Green":
        return [
            "Perform monthly self-checks",
            "Maintain healthy diet and exercise",
            "Monitor for new symptoms"
        ]
    elif band == "Yellow":
        return [
            "Track symptoms for 7–14 days",
            "Schedule screening if symptoms persist",
            "Improve sleep, reduce stress, healthy lifestyle"
        ]
    return [
        "Strongly recommend clinic screening",
        "Book medical appointment as soon as possible",
        "Prepare symptom history for doctor visit"
    ]

def predict(data: dict):
    # Convert dict to DataFrame (required by ColumnTransformer in the pipeline)
    df = pd.DataFrame([data])
    
    # Get prediction probability
    prob = model.predict_proba(df)[0][1]
    band = risk_band(prob)

    return {
        "risk_band": band,
        "probability": round(float(prob), 4),
        "recommendations": recommendations(band),
        "disclaimer": "This is not a diagnosis. Please consult a healthcare professional."
    }
