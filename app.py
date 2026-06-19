import os
import joblib
import pandas as pd
import numpy as np
import shap
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from groq import Groq

# Configure logging to securely track internal errors without leaking them to users
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bankmind-api")

app = FastAPI(
    title="AuditMind Backend Service",
    description="AI-Driven Cross-Sell Recommendation Core Engine with SHAP Explainability",
    version="1.0.0"
)

# --- GLOBAL MODEL & SHAP INITIALIZATION ---
MODEL_PATH = "model.pkl"
if os.path.exists(MODEL_PATH):
    model_pipeline = joblib.load(MODEL_PATH)
    preprocessor = model_pipeline.named_steps['preprocessor']
    classifier = model_pipeline.named_steps['classifier']
    
    # Initialize Explainer ONCE globally for maximum performance optimization
    shap_explainer = shap.TreeExplainer(classifier)
else:
    raise RuntimeError(f"Serialization asset '{MODEL_PATH}' not found. Run train.py first.")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# --- SCHEMAS (With Input Validation & Contract Alignment) ---
class CustomerData(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Age of the customer")
    job: str = Field(..., description="Type of job")
    marital: str = Field(..., description="Marital status")
    education: str = Field(..., description="Education level")
    default: str = Field("no", description="Has credit in default?")
    balance: int = Field(..., description="Average yearly balance in euros")
    housing: str = Field(..., description="Has housing loan?")
    loan: str = Field(..., description="Has personal loan?")
    contact: str = Field("unknown", description="Contact communication type")
    day: int = Field(15, ge=1, le=31, description="Last contact day of the month")
    month: str = Field("may", description="Last contact month of year")
    duration: int = Field(0, ge=0, description="Last contact duration in seconds")
    campaign: int = Field(1, ge=1, description="Number of contacts performed during this campaign")
    pdays: int = Field(-1, description="Number of days that passed by after the customer was last contacted")
    previous: int = Field(0, ge=0, description="Number of contacts performed before this campaign")
    poutcome: str = Field("unknown", description="Outcome of the previous marketing campaign")

class PredictionResponse(BaseModel):
    will_subscribe: bool
    probability: float = Field(..., description="Decimal probability matching the exact assignment contract")
    probability_percentage: str = Field(..., description="User-friendly percentage string for Relationship Managers")
    top_factors: list[str]

class ExplanationRequest(BaseModel):
    customer: CustomerData
    probability: float = Field(..., description="Numeric decimal probability for calculation clarity")

# --- SHAP NATURAL TEXT CONVERSION ENGINE ---
def get_human_readable_shap(input_df: pd.DataFrame) -> list:
    try:
        X_transformed = preprocessor.transform(input_df)
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()
            
        feature_names = [name.split('__')[-1] for name in preprocessor.get_feature_names_out()]
        shap_values = shap_explainer.shap_values(X_transformed)
        
        if isinstance(shap_values, list):
            vals = shap_values[1][0]
        elif len(np.array(shap_values).shape) == 3:
            vals = shap_values[0, :, 1]
        else:
            vals = shap_values[0]
            
        vals = np.array(vals).flatten()
        impacts = list(zip(feature_names, vals))
        impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_factors = []
        for feat, val in impacts[:3]:
            direction = "increased" if val > 0 else "decreased"
            
            # Transform raw columns into highly polished, conversational sentences
            if feat == "duration":
                phrase = f"A longer call duration {direction}"
            elif feat == "balance":
                phrase = f"A higher account balance {direction}"
            elif feat == "housing_no":
                phrase = f"The lack of an existing housing loan {direction}"
            elif feat == "housing_yes":
                phrase = f"An active housing loan debt {direction}"
            elif feat == "loan_no":
                phrase = f"The absence of personal loan liabilities {direction}"
            elif feat == "loan_yes":
                phrase = f"An active personal loan {direction}"
            elif "poutcome_success" in feat:
                phrase = f"The successful outcome of the previous campaign {direction}"
            else:
                clean_feat = feat.replace('_', ' ').title()
                phrase = f"The customer attribute '{clean_feat}' {direction}"
                
            top_factors.append(f"{phrase} the likelihood of subscription.")
            
        return top_factors
    except Exception as e:
        logger.error(f"SHAP parsing warning: {str(e)}")
        return ["Baseline profile indicators drove this prediction.", "Standard demographic traits matched."]

# --- ENDPOINTS ---
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": "Random Forest Classifier Pipeline",
        "version": "1.0.0",
        "shap_engine": "active",
        "groq_connected": groq_client is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_subscription(customer: CustomerData):
    try:
        input_data = pd.DataFrame([customer.model_dump()])
        prediction = int(model_pipeline.predict(input_data)[0])
        probability_val = float(model_pipeline.predict_proba(input_data)[0][1])
        
        factors = get_human_readable_shap(input_data)
        
        return {
            "will_subscribe": True if prediction == 1 else False,
            "probability": round(probability_val, 4),
            "probability_percentage": f"{probability_val * 100:.2f}%",
            "top_factors": factors
        }
    except Exception as e:
        logger.error(f"Inference processing failure: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed. Please verify the structural format of your input data.")

@app.post("/explain")
def explain_prediction(req: ExplanationRequest):
    if not groq_client:
        raise HTTPException(status_code=501, detail="Groq Client missing. Set GROQ_API_KEY environment variable.")
    
    try:
        cust = req.customer
        prob_pct = req.probability * 100
        
        prompt = f"""Customer profile:
- Age: {cust.age}, Job: {cust.job}, Balance: €{cust.balance}
- Existing loans: Housing={cust.housing}, Personal={cust.loan}
- Model prediction: {prob_pct:.2f}% chance of subscribing to a term deposit

In 2-3 sentences, explain why this customer would or would not likely
subscribe to a term deposit, and how an RM should approach the conversation."""

        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=150
        )
        return {"explanation": chat_completion.choices[0].message.content.strip()}
    except Exception as e:
        logger.error(f"GenAI LLM generation failure: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Explanation generation failed. Internal LLM engine encounter.")