# 🏦 BankMind – Intelligent Cross-Sell Recommendation Service

An AI-powered banking recommendation system that predicts the probability of a customer subscribing to a banking product and provides explainable AI insights using Machine Learning, SHAP Explainability, and Large Language Models (LLMs).

Built as a submission for **Track C – System Builder** under the **AuditMind: AI-Driven Cross-Sell Recommendation System** screening project.

---

# 🚀 Features

* 🤖 Machine Learning-based subscription prediction
* 📊 Probability score generation
* 🔍 SHAP-powered feature attribution
* 🧠 LLM-based business explanations using Groq
* ⚡ FastAPI REST API
* 📈 Automated model training pipeline
* 📦 Serialized model deployment artifact (`model.pkl`)
* 📑 Interactive Swagger API documentation
* 🛡️ Health monitoring endpoint
* 📋 Feature importance extraction

---

# 🛠️ Tech Stack

* Python 3.10+
* FastAPI
* Scikit-learn
* Pandas
* NumPy
* Joblib
* SHAP
* Uvicorn
* Groq API

---

# 📁 Project Structure

```text
bankmind-mohitpal/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── EXPLANATION.md
├── model.pkl
├── feature_importance.csv
└── bank-full.csv
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/MohitPal2005/bankmind-mohitpal.git

cd bankmind-mohitpal
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🧠 Train the Model

Run the training pipeline:

```bash
python train.py
```

This will:

* Download the UCI Bank Marketing Dataset (if not present)
* Perform focused EDA
* Train Logistic Regression and Random Forest models
* Compare performance metrics
* Generate feature importance rankings
* Save the final model as `model.pkl`

Generated files:

```text
model.pkl
feature_importance.csv
```

---

# 📊 Model Performance

## Logistic Regression

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 0.9012 |
| Precision | 0.6445 |
| Recall    | 0.3478 |
| F1 Score  | 0.4518 |

## Random Forest

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 0.9059 |
| Precision | 0.6987 |
| Recall    | 0.3440 |
| F1 Score  | 0.4611 |

### Selected Model

🏆 Random Forest Classifier

Chosen because it achieved the highest F1-score and supports SHAP-based explainability.

---

# 🔑 Configure Environment Variables

The explainability endpoint uses the Groq API.

### Windows PowerShell

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
```

### Linux / macOS

```bash
export GROQ_API_KEY="your_groq_api_key"
```

---

# ▶️ Run the API

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Server URL:

```text
http://127.0.0.1:8000
```

---

# 📖 API Documentation

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

ReDoc:

```text
http://127.0.0.1:8000/redoc
```

---

# 🔍 API Endpoints

| Method | Endpoint | Description                   |
| ------ | -------- | ----------------------------- |
| GET    | /health  | Service health check          |
| POST   | /predict | Predict customer subscription |
| POST   | /explain | Generate AI explanation       |

---

# 🧪 Example API Usage

## 1. Health Check

### Request

```bash
curl -X GET \
"http://127.0.0.1:8000/health" \
-H "accept: application/json"
```

### Example Response

```json
{
  "status": "ok",
  "model": "Random Forest Classifier Pipeline",
  "version": "1.0.0",
  "shap_engine": "active",
  "groq_connected": true
}
```

---

## 2. Prediction Endpoint

### Request

```bash
curl -X POST \
"http://127.0.0.1:8000/predict" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d '{
  "age":64,
  "job":"retired",
  "marital":"married",
  "education":"secondary",
  "default":"no",
  "balance":109,
  "housing":"no",
  "loan":"no",
  "contact":"cellular",
  "day":15,
  "month":"jun",
  "duration":450,
  "campaign":1,
  "pdays":-1,
  "previous":0,
  "poutcome":"unknown"
}'
```

### Example Response

```json
{
  "will_subscribe": true,
  "probability": 0.6067,
  "probability_percentage": "60.67%",
  "top_factors": [
    "A longer call duration increased the likelihood of subscription.",
    "A higher account balance decreased the likelihood of subscription.",
    "The customer attribute 'Age' increased the likelihood of subscription."
  ]
}
```

---

## 3. Explainability Endpoint

### Request

```bash
curl -X POST \
"http://127.0.0.1:8000/explain" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d '{
  "customer":{
    "age":64,
    "job":"retired",
    "marital":"married",
    "education":"secondary",
    "default":"no",
    "balance":109,
    "housing":"no",
    "loan":"no",
    "contact":"cellular",
    "day":15,
    "month":"jun",
    "duration":450,
    "campaign":1,
    "pdays":-1,
    "previous":0,
    "poutcome":"unknown"
  },
  "probability":0.6067
}'
```

### Example Response

```json
{
  "explanation": "This customer is likely to subscribe to a term deposit due to their retired status and preference for low-risk investment options. The Relationship Manager should focus on fixed-return savings benefits and align the conversation with the customer's financial goals."
}
```

---

# 🔍 SHAP Explainability

The prediction service uses SHAP (SHapley Additive exPlanations) to identify the most influential features behind every prediction.

Examples:

* Call duration
* Account balance
* Age
* Housing loan status
* Previous campaign outcome

This enables transparent and explainable recommendations instead of black-box predictions.

---

# 📈 Top Features Learned by the Model

According to the trained Random Forest model:

| Rank | Feature  |
| ---- | -------- |
| 1    | duration |
| 2    | balance  |
| 3    | age      |
| 4    | day      |
| 5    | campaign |

A complete ranking is exported to:

```text
feature_importance.csv
```

---

# 📊 Workflow

```text
Customer Data
       │
       ▼
Preprocessing Pipeline
       │
       ▼
Random Forest Model
       │
       ▼
Probability Prediction
       │
       ▼
SHAP Feature Attribution
       │
       ▼
Groq LLM Explanation
       │
       ▼
Business Recommendation
```

---

# 📌 Future Improvements

* Interactive SHAP Dashboard
* Batch Prediction Endpoint
* Docker Containerization
* CI/CD Pipeline
* Authentication & Authorization
* Cloud Deployment
* Model Monitoring
* Database Integration

---

# 👨‍💻 Author

**Mohit Pal**

B.Tech Computer Science Engineering

VIT Bhopal University

Track C – System Builder Submission
