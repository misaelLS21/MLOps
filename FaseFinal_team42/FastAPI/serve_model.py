from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn._loss import HalfBinomialLoss

# Initialize FastAPI app
app = FastAPI(title="Model Serving API", description="API for serving a machine learning model", version="1.0.0")

# Define input schema using Pydantic
class PredictionInput(BaseModel):
    features: List[float]

# Define output schema using Pydantic
class PredictionOutput(BaseModel):
    prediction: int
    probability: float

# Custom loader for compatibility
def custom_loader():
    return GradientBoostingClassifier()

# Define a custom mapping for missing attributes
custom_objects = {
    '__pyx_unpickle_CyHalfBinomialLoss': HalfBinomialLoss
}

# Load the model artifact
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/final_model_GradBoost.joblib")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")

# Use a custom unpickler to handle compatibility
with open(MODEL_PATH, "rb") as model_file:
    model = joblib.load(model_file, mmap_mode=None)

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """Endpoint to make predictions using the loaded model."""
    try:
        # Ensure the input features are valid
        if len(input_data.features) != model.n_features_in_:
            raise HTTPException(status_code=400, detail=f"Expected {model.n_features_in_} features, got {len(input_data.features)}")

        # Make prediction
        prediction = model.predict([input_data.features])[0]
        probability = model.predict_proba([input_data.features])[0].max()

        return PredictionOutput(prediction=int(prediction), probability=float(probability))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")
