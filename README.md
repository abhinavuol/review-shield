# ReviewShield

### ML-Powered Fake Review Detection API

ReviewShield is a machine learning service that detects potentially deceptive product reviews using Natural Language Processing (NLP). The system trains a classification model on a real benchmark dataset and exposes predictions through a REST API built with FastAPI.

This project demonstrates an end-to-end machine learning workflow including dataset preparation, model training, evaluation, model serialization, and API-based model serving.

---

# 🚀 Features

* Fake vs Genuine review classification
* NLP pipeline using TF-IDF and Logistic Regression
* Real research dataset integration
* REST API for programmatic inference
* Browser interface for manual testing
* Automatic API documentation with FastAPI
* Modular training and inference architecture

---

# 🧠 Model Overview

The model uses a traditional NLP pipeline:

Text Review
→ TF-IDF Vectorization
→ Logistic Regression Classifier
→ Fake/Genuine Prediction

This pipeline allows efficient text classification with interpretable word-based features.

---

# 📊 Dataset

The model is trained using the **Ott Deceptive Opinion Spam Dataset**, a benchmark dataset used in academic research for deceptive review detection.

Dataset properties:

* 1600 hotel reviews
* 800 deceptive reviews
* 800 truthful reviews
* Balanced dataset
* Positive and negative sentiment reviews

Dataset source:
https://myleott.com/op_spam_v1.4.zip

The raw dataset consists of text files organized in directories.
A preprocessing script converts these files into a structured CSV dataset for training.

---

# 📈 Model Performance

Evaluation results on a held-out test set:

| Metric   | Score     |
| -------- | --------- |
| Accuracy | **87.5%** |
| F1 Score | **0.874** |

Confusion Matrix:

|                | Predicted Genuine | Predicted Fake |
| -------------- | ----------------- | -------------- |
| Actual Genuine | 141               | 19             |
| Actual Fake    | 21                | 139            |

The model performs well with balanced precision and recall across both classes.

---

# 🏗 System Architecture

User / Application
↓
FastAPI Backend
↓
ReviewShield Predictor
↓
Saved ML Pipeline
↓
TF-IDF + Logistic Regression

The model is trained offline and loaded at runtime for fast predictions.

---

# 📁 Project Structure

```
reviewshield/
│
├── app/
│   ├── main.py
│   ├── predictor.py
│   └── schemas.py
│
├── model/
│   ├── train.py
│   ├── test_model.py
│   ├── pipeline.pkl
│   └── metrics.json
│
├── data/
│   ├── prepare_dataset.py
│   ├── raw/
│   └── reviews.csv
│
├── templates/
│   └── index.html
│
├── static/
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Running the Project

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python model/train.py
```

### Start the API server

```bash
uvicorn app.main:app --reload
```

Then open:

```
http://127.0.0.1:8000
```

API documentation:

```
http://127.0.0.1:8000/docs
```

---

# 🔎 Example API Request

POST `/predict`

Request:

```json
{
"text": "This product is absolutely amazing, best purchase ever!"
}
```

Response:

```json
{
"label": "fake",
"fake_probability": 0.87
}
```

---

# 🖥 Example UI

The project also includes a simple web interface for manual testing.

Users can paste a review and instantly see the predicted label and probability.

---

# 🧪 Future Improvements

Possible extensions include:

* Transformer-based models (BERT)
* Larger review datasets
* Model explainability for suspicious words
* Docker containerization
* Deployment on Raspberry Pi or cloud infrastructure
* Rate limiting and authentication for the API

---

# 📌 Why This Project

This project demonstrates practical machine learning system design including:

* NLP model development
* dataset preprocessing pipelines
* model serialization
* API-based ML deployment
* modular project structure

These are common patterns used when deploying machine learning models in production environments.

---

# 📜 License

This project is intended for educational and portfolio purposes.
