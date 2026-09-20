"""
Prediction utilities for the Airbnb India price model.

Loads the trained pipeline (preprocessing + regressor) saved by
src/train.py and turns raw listing details into a price prediction.
No preprocessing logic lives here -- it's all inside the saved pipeline.
"""
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"


def load_model(model_path: Path = MODEL_PATH):
    """Load the trained pipeline from disk.

    Raises FileNotFoundError with a clear message if training hasn't been
    run yet, so callers (like the Streamlit app) can catch it and show a
    friendly message instead of a traceback.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run `python main.py` first."
        )
    return joblib.load(model_path)


def predict_price(input_data: dict, model=None, model_path: Path = MODEL_PATH) -> float:
    """Predict the nightly price for a single listing.

    Parameters
    ----------
    input_data : dict
        Feature name -> value, e.g. {"state": "Goa", "propertyType": "Villa",
        "numberOfGuests": 4, ...}. Keys should match the feature columns the
        model was trained on.
    model : fitted sklearn Pipeline, optional
        Pass an already-loaded pipeline (e.g. cached in Streamlit) to
        avoid reloading it from disk on every call.
    model_path : Path
        Where to load the model from if `model` isn't provided.

    Returns
    -------
    float
        Predicted nightly price (in rupees).
    """
    if model is None:
        model = load_model(model_path)

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    return float(prediction)
