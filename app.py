from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

app = Flask(__name__)

# Load dataset
DATA_PATH = './uploads/Training_Dataset.csv'

def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)

    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Encode categorical variables
    target = 'Depression_State'
    X = df.drop(columns=[target])
    y = df[target]

    # Encode the target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Save the encoder for mapping predictions back
    joblib.dump(label_encoder, 'label_encoder.pkl')

    # Encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X[categorical_cols]).toarray()

    X = X.drop(columns=categorical_cols)
    X = np.hstack([X.values, X_encoded])

    # Scale numerical features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return X, y, encoder, scaler

X, y, encoder, scaler = load_and_preprocess_data()

# Train models
models = {
    #'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    #'LightGBM': LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=10),
    'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    #'LogisticRegression': LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
}

for name, model in models.items():
    model.fit(X, y)
    joblib.dump(model, f'{name}.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse form data
        form_data = request.form.to_dict()
        input_data = pd.DataFrame([form_data])

        # Convert all inputs to numeric
        input_data = input_data.apply(pd.to_numeric, errors='coerce')

        # Preprocess form data
        categorical_cols = input_data.select_dtypes(include=['object']).columns
        X_encoded = encoder.transform(input_data[categorical_cols]).toarray()
        input_data = input_data.drop(columns=categorical_cols, errors='ignore')
        input_data = np.hstack([input_data.values, X_encoded])
        input_data = scaler.transform(input_data)

        results = {}

        # Predict using trained models
        for name, model in models.items():
            model = joblib.load(f'{name}.pkl')
            predictions = model.predict(input_data)

            # Decode predictions back to original labels
            label_encoder = joblib.load('label_encoder.pkl')
            decoded_predictions = label_encoder.inverse_transform(predictions)

            results[name] = decoded_predictions[0]  # Get the first prediction

        return render_template('predict.html', results=results)

    except Exception as e:
        return render_template('predict.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
