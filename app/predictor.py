from pathlib import Path

import joblib
import numpy as np


MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "pipeline.pkl"


class ReviewPredictor:
    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

        self.pipeline = joblib.load(MODEL_PATH)
        self.vectorizer = self.pipeline.named_steps["tfidf"]
        self.classifier = self.pipeline.named_steps["clf"]

        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        self.coefficients = self.classifier.coef_[0]

    def explain(self, text: str, top_k: int = 5) -> list[str]:
        """
        Return top words/phrases in the input that contribute most
        toward the fake class prediction.
        """
        X = self.vectorizer.transform([text])
        row = X[0]

        if row.nnz == 0:
            return []

        indices = row.indices
        values = row.data

        contributions = []
        for idx, tfidf_value in zip(indices, values):
            coef = self.coefficients[idx]
            contribution = tfidf_value * coef

            if contribution > 0:
                contributions.append((self.feature_names[idx], contribution))

        contributions.sort(key=lambda x: x[1], reverse=True)
        return [feature for feature, _ in contributions[:top_k]]

    def predict(self, text: str) -> tuple[str, float, list[str]]:
        probabilities = self.pipeline.predict_proba([text])[0]
        prediction = self.pipeline.predict([text])[0]

        fake_probability = float(probabilities[1])
        label = "fake" if int(prediction) == 1 else "genuine"
        suspicious_signals = self.explain(text)

        return label, fake_probability, suspicious_signals