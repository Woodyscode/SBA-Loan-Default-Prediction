BA Loan Default Prediction Dashboard

An interactive Streamlit dashboard to predict SBA loan default risk using models trained on real vs. real+synthetic features.

Features

Load and clean SBA national loan data

Train RandomForest models on real and synthetic features

Predict default risk from structured form or uploaded PDFs

Visualizations: SHAP, calibration, ROC, PSI, feature importances

Upload real or fake financial PDFs for document parsing and prediction

Files

Train_model.py: Trains models and saves .pkl files

dashboard.py: Streamlit app with full UI

SBAnational.csv: Cleaned SBA loan data

loan_model_real.pkl: Model trained on real features

loan_model_synth.pkl: Model trained on real + synthetic features

Setup

# Create virtual env and install dependencies
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard.py

Example

Upload a PDF loan summary or manually enter values to get default risk predictions in real time.

