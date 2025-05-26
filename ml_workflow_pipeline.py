from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import joblib
from datetime import datetime

app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None
columns = ['age', 'income', 'credit_score', 'employment_years', 'loan_amount', 'loan_term']
feature_columns = [
    'age', 
    'income', 
    'credit_score', 
    'employment_years', 
    'loan_amount',
    'debt_to_income',
    'credit_score_bin_poor',
    'credit_score_bin_fair',
    'credit_score_bin_good',
    'credit_score_bin_excellent',
    'loan_term_12',
    'loan_term_24',
    'loan_term_36',
    'loan_term_60'
]
DATA_FROM_CLOUD = True


def read_data_from_cloud():
    response = requests.get('https://api.jsonbin.io/v3/b/68323e288a456b7966a4ee00')
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return False

def read_data_from_local():
    with open('ml_data_compact.json') as f:
        compact_data = json.load(f)
    return compact_data

def read_data():
    if DATA_FROM_CLOUD:
        compact_data = read_data_from_cloud()
    else:
        compact_data = read_data_from_local()
    
    # Convert to DataFrame
    import pandas as pd
    df = pd.DataFrame(compact_data.get('record').get('data'), columns=compact_data.get('record').get('columns'))
    # Convert timestamp back to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    return df

def clean_data(df):
    """Handle data issues in the raw data"""
    # Handle missing values
    df['age'].fillna(df['age'].median(), inplace=True)
    
    # Fix type inconsistencies
    df['credit_score'] = pd.to_numeric(df['credit_score'], errors='coerce')
    df['credit_score'].fillna(df['credit_score'].median(), inplace=True)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Cap outliers (income > 200k set to 200k)
    df['income'] = np.where(df['income'] > 200000, 200000, df['income'])
    
    return df

def generate_features(df):
    """Create additional features"""
    # Ratio features
    df['debt_to_income'] = df['loan_amount'] / df['income']
    
    # Binning
    df['credit_score_bin'] = pd.cut(df['credit_score'], 
                                   bins=[0, 580, 670, 740, 850],
                                   labels=['poor', 'fair', 'good', 'excellent'])
    
    # One-hot encoding
    df = pd.get_dummies(df, columns=['credit_score_bin', 'loan_term'])
    
    return df

def train_model():
    """Train and save the model"""
    global model, scaler
    
    # Load and prepare data
    df = read_data()
    
    # df = pd.DataFrame(data)
    df = clean_data(df)
    df = generate_features(df)
    
    # Select final features
    X = df[feature_columns]
    y = df['approved']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train simple model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Save artifacts
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    # Return training metrics
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(scaler.transform(X_test), y_test)
    
    return {
        "status": "success",
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "features_used": feature_columns,
        "timestamp": datetime.now().isoformat()
    }

@app.route('/predict', methods=['GET'])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model not trained. First call /train"}), 400
    
    try:
        # Get and validate parameters
        age = float(request.args.get('age', 30))
        income = float(request.args.get('income', 50000))
        credit_score = float(request.args.get('credit_score', 700))
        employment_years = float(request.args.get('employment_years', 5))
        loan_amount = float(request.args.get('loan_amount', 20000))
        loan_term = int(request.args.get('loan_term', 36))
        
        if loan_term not in [12, 24, 36, 60]:
            return jsonify({"error": "loan_term must be one of: 12, 24, 36, 60"}), 400
        
        # Create base features
        features = {
            'age': age,
            'income': income,
            'credit_score': credit_score,
            'employment_years': employment_years,
            'loan_amount': loan_amount,
            'debt_to_income': loan_amount / max(income, 1),  # prevent division by zero
        }
        
        # Add credit score bins (one-hot encoded)
        bins = {
            'credit_score_bin_poor': (0, 580),
            'credit_score_bin_fair': (580, 670),
            'credit_score_bin_good': (670, 740),
            'credit_score_bin_excellent': (740, 850)
        }
        
        for col, (min_val, max_val) in bins.items():
            features[col] = 1 if min_val <= credit_score < max_val else 0
        
        # Add loan term (one-hot encoded)
        for term in [12, 24, 36, 60]:
            features[f'loan_term_{term}'] = 1 if term == loan_term else 0
        
        # Create DataFrame with EXACTLY the right columns in the right order
        input_df = pd.DataFrame([features])[feature_columns]
        
        # Scale and predict
        scaled_input = scaler.transform(input_df)
        proba = model.predict_proba(scaled_input)[0]
        prediction = model.predict(scaled_input)[0]
        
        return jsonify({
            "prediction": int(prediction),
            "probability_approved": float(proba[1]),
            "probability_denied": float(proba[0]),
            "input_features": features,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/train', methods=['GET'])
def train():
    """Training endpoint"""
    result = train_model()
    return jsonify(result)

if __name__ == '__main__':
    # Train model when starting the server
    train_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
