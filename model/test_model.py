from pathlib import Path
import joblib


def main() -> None:
    model_path = Path(__file__).resolve().parent / "pipeline.pkl"
    pipeline = joblib.load(model_path)

    sample_reviews = [
        "Absolutely incredible product, everyone should buy this right now.",
        "Arrived damaged and stopped working after one day.",
        "Five stars, best purchase of my life, unbelievable value.",
        "The product was okay, but delivery was slow."
    ]

    predictions = pipeline.predict(sample_reviews)
    probabilities = pipeline.predict_proba(sample_reviews)

    for review, pred, prob in zip(sample_reviews, predictions, probabilities):
        fake_prob = prob[1]
        print("-" * 80)
        print(f"Review: {review}")
        print(f"Predicted label: {pred}")
        print(f"Fake probability: {fake_prob:.4f}")


if __name__ == "__main__":
    main()