from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import joblib


def main() -> None:
	#Define the paths
	project_root = Path(__file__).resolve().parents[1]
	data_path = project_root / "data" / "reviews.csv"
	output_model_path = Path(__file__).resolve().parent / "pipeline.pkl"
	output_metrics_path = Path(__file__).resolve().parent /"metrics.json"

	#Load Dataset
	df = pd.read_csv(data_path)

	#Basic visualization
	required_columns = {"text", "label"}
	if not required_columns.issubset(df.columns):
		raise ValueError(f"CSV must contain columns: {required_columns}")

	
	df = df.dropna(subset=["text", "label"])

	#Training variables

	X = df["text"].astype(str)
	y = df["label"].astype(int)

	#Train/test split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

	#Pipeline
	pipeline = Pipeline(
		steps=[
			("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=2)),
			("clf", LogisticRegression(max_iter=2000))

		]

	)

	#Training
	pipeline.fit(X_train, y_train)

	#Evaluation
	y_pred = pipeline.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	report = classification_report(y_test, y_pred, output_dict=True)
	cm = confusion_matrix(y_test, y_pred)
	f1 = f1_score(y_test, y_pred)

	metrics = {
		"accuracy": round(float(accuracy),4),
		"f1_score": round(float(f1), 4),
		"classification_report": report,
		"confusion_matrix":cm.tolist(),
		"train_size": int(len(X_train)),
		"test_size": int(len(X_test))
	}

	#Save model and metrics
	joblib.dump(pipeline, output_model_path)

	with open(output_metrics_path, "w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)

	print("Training complete")
	print(f"Model saved to: {output_model_path}")
	print(f"Metrics saved to: {output_metrics_path}")
	print(f"Accuracy: {accuracy:.4f}")
	print(f"F1 Score: {f1:.4f}")
	print("Confusion Matrix:")
	print(cm)

if __name__ == "__main__":
	main()