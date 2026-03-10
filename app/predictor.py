from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "pipeline.pkl"

class ReviewPredictor:
	def __init__(self) -> None:
		if not MODEL_PATH.exists():
			raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
		self.pipeline = joblib.load(MODEL_PATH)

	def predict(self, text: str) -> tuple[str, float]:
		probabilities = self.pipeline.predict_proba([text])[0]
		prediction = self.pipeline.predict([text])[0]

		fake_probability = float(probabilities[1])
		label = "fake" if int(prediction) == 1 else "Genuine"

		return label, fake_probability
		