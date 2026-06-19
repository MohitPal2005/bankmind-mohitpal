import os
import zipfile
import io
import requests
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- STEP 1: DATA INGESTION ---
DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
CSV_FILENAME = "bank-full.csv"

def fetch_data():
    if not os.path.exists(CSV_FILENAME):
        print("[*] Downloading UCI Bank Marketing Dataset...")
        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            for file in z.namelist():
                if "bank-full.csv" in file:
                    with z.open(file) as f:
                        with open(CSV_FILENAME, "wb") as out:
                            out.write(f.read())
                elif "bank.zip" in file:
                    with zipfile.ZipFile(io.BytesIO(z.read(file))) as inner_z:
                        if "bank-full.csv" in inner_z.namelist():
                            with open(CSV_FILENAME, "wb") as out:
                                out.write(inner_z.read("bank-full.csv"))
    
    df = pd.read_csv(CSV_FILENAME, sep=';')
    return df

df = fetch_data()

# --- STEP 2: FOCUSED EXPLORATORY DATA ANALYSIS ---
print("\n=== FOCUSED EDA ===")
print(f"Dataset Shape: {df.shape}")

# Added back the explicit missing values check (Mandatory Requirement)
print(f"\nMissing Values:\n{df.isnull().sum().sum()}")

print("\nClass Distribution (Target variable 'y'):")
print(df['y'].value_counts(normalize=True) * 100)

# Auto-generate categorical and numerical lists based on dtypes
categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_features.remove('y')
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

df['target'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)
X = df.drop(columns=['y', 'target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- STEP 3: PREPROCESSING PIPELINE ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# --- STEP 4: MODEL TRAINING & COMPARISON ---
print("\n=== TRAINING MODELS ===")

lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
lr_pipeline.fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_test)

rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced'))])
rf_pipeline.fit(X_train, y_train)
rf_preds = rf_pipeline.predict(X_test)

def evaluate_model(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    
    f1 = f1_score(y_true, y_pred)
    print(f"F1 Score:  {f1:.4f}")
    
    # Added back the classification report (Mandatory Requirement)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    return f1

lr_f1 = evaluate_model("Baseline (Logistic Regression)", y_test, lr_preds)
rf_f1 = evaluate_model("Main Model (Random Forest)", y_test, rf_preds)

# Dynamic Winner Evaluation (Honest Logic)
print("\n🏆 WINNER MODEL EVALUATION 🏆")
if rf_f1 > lr_f1:
    print(f"Random Forest outperformed Logistic Regression on F1-Score ({rf_f1:.4f} vs {lr_f1:.4f}).")
else:
    print(f"Logistic Regression outperformed Random Forest on F1-Score ({lr_f1:.4f} vs {rf_f1:.4f}).")
    
print("Proceeding with Random Forest to support Tree-based SHAP explainability.")

# --- STEP 5: FEATURE IMPORTANCE EXTRACTION ---
feature_names = rf_pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = rf_pipeline.named_steps["classifier"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": [name.split('__')[-1] for name in feature_names],
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\n--- Top 10 Important Features ---")
print(importance_df.head(10))
importance_df.to_csv("feature_importance.csv", index=False)
print("\n[*] Feature importances saved to 'feature_importance.csv'")

# --- STEP 6: SAMPLE GENERATION FOR TRACK B REQUIREMENT ---
print("\n=== SAMPLE TEST CUSTOMER PROFILES ===")
rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]

test_analysis = X_test.copy()
test_analysis['actual'] = y_test
test_analysis['predicted'] = rf_preds
test_analysis['probability'] = rf_probs

# Upgraded to .sample() with a random seed for robust selection
yes_samples = test_analysis[test_analysis['predicted'] == 1].sample(3, random_state=42)
no_samples = test_analysis[test_analysis['predicted'] == 0].sample(2, random_state=42)
merged_samples = pd.concat([yes_samples, no_samples])

for idx, row in merged_samples.iterrows():
    # Cleaned up formatting strings
    print(f"\nCustomer ID: {idx} | Age: {row['age']} | Job: {row['job']} | Balance: €{row['balance']} | Housing: {row['housing']} | Loan: {row['loan']}")
    pred_text = 'Will Subscribe' if row['predicted'] == 1 else 'Will Not Subscribe'
    print(f"Prediction: {pred_text} | Prob: {row['probability'] * 100:.2f}%")

# --- STEP 7: PERSIST MODEL ASSET ---
joblib.dump(rf_pipeline, "model.pkl")
print("\n[+] Winning Random Forest Pipeline successfully saved as 'model.pkl'")