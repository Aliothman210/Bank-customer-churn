import streamlit as st
import pickle as pkl
import numpy as np
import time
import xgboost
from sklearn.ensemble import GradientBoostingClassifier 
import base64

# Load models
models = {
    'logistic': pkl.load(open('logistic.pkl', 'rb')),
    'Ada': pkl.load(open('Ada.pkl', 'rb')),
    
   
    'svm': pkl.load(open('svm.pkl', 'rb')),
    'XG': pkl.load(open('XG.pkl', 'rb')),
    'decision_tree': pkl.load(open('decision_tree.pkl', 'rb')),
}

models_names = list(models.keys())

# Load scaler and transformer
scaler = pkl.load(open('robustscaler.pkl', 'rb'))
power = pkl.load(open('powertransformer.pkl', 'rb'))

# Define feature lists
input_features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
col1_features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure']
col2_features = ['Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
cat_features = ['Geography', 'Gender']
Yes_no_features = ['HasCrCard', 'IsActiveMember']

# Function to load encoders for categorical features
def load_encoder(feature):
    return pkl.load(open(f'{feature}.pkl', 'rb'))

# Main function for Streamlit app
def main():
    st.title(":bank: Bank Churn Prediction")
    
    # Create columns for input fields
    col1, col2 = st.columns(2)
    inputs = {}

    # Collect inputs for features in col1
    for feature in col1_features:
        with col1:
            if feature in cat_features:
                en = load_encoder(feature)
                options = en.categories_[0].tolist() if hasattr(en, 'categories_') else en.classes_.tolist()
                inputs[feature] = st.selectbox(feature, options)
            else:
                inputs[feature] = st.number_input(feature)

    # Collect inputs for features in col2
    for feature in col2_features:
        with col2:
            if feature in cat_features:
                en = load_encoder(feature)
                options = en.categories_[0].tolist() if hasattr(en, 'categories_') else en.classes_.tolist()
                inputs[feature] = st.selectbox(feature, options)
            elif feature in Yes_no_features:
                choice = st.selectbox(feature, ['Yes', 'No'])
                inputs[feature] = 1 if choice == 'Yes' else 0
            else:
                inputs[feature] = st.number_input(feature)
    
    # Model selection
    model_choice = st.selectbox("Choose Model", models_names)
    model = models[model_choice]

    # Prediction
    if st.button('Predict'):
        with st.spinner('Making prediction...'):
            time.sleep(0.8)
            features = []

            # Encode categorical features and prepare inputs
            for feature in input_features:
                value = inputs[feature]
                if feature in cat_features:
                    en = load_encoder(feature)
                    value = en.transform(np.array([[value]]))[0]
                features.append(value)

            features = np.array(features, dtype='object').reshape(1, -1)
            features_scaled = power.transform(features)
            features_scaled = scaler.transform(features_scaled)

            # Check for monotonic_cst attribute to avoid errors
            if not hasattr(model, "monotonic_cst"):
                model.monotonic_cst = None

            # Make prediction
            y_pred = model.predict(features_scaled)
            if y_pred == 1:
                st.error('The customer is likely to churn.')
            else:
                st.success('The customer is likely to stay.')

if __name__ == '__main__':
    main()
