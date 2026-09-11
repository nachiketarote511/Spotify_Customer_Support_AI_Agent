"""
Intent Prediction Module
Wraps the trained classifier for easy inference.
"""

from app.intent.train import IntentPredictor, IntentClassifier


def predict_intent(message: str, model_dir: str = None, model_name: str = 'baseline') -> dict:
    """
    Predict intent for a customer message.
    
    Returns:
        dict with keys: intent, confidence, all_probabilities
    """
    predictor = IntentPredictor(model_dir, model_name)
    return predictor.predict(message)
