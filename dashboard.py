#!/usr/bin/env python3
# streamlit_dashboard.py

"""
Real-time SBA Loan Default Risk Dashboard (Enhanced)
Compares models trained on real-only vs. real+synthetic features.
Tabs: Statistics, EDA, Predict, Compare, Health, Info.
"""

# ─── IMPORTS ──────────────────────────────────────────────────────────────────
import streamlit as st
import fitz
import pandas as pd
import numpy as np
import joblib
import tempfile
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from faker import Faker
import os
import re
import shap
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pytesseract, pdf2image, shap
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
DATA_PATH        = "SBAnational.csv"
MODEL_REAL_PATH  = "loan_model_real.pkl"
MODEL_SYNTH_PATH = "loan_model_synth.pkl"

FEATURES_REAL = [
    'Term',
    'NoEmp',
    'NewExist',
    'CreateJob',
    'RetainedJob',
    'FranchiseCode',
    'UrbanRural',
    'RevLineCr',
    'LowDoc',
    'DisbursementGross',
    'BalanceGross',
    'GrAppv',
    'SBA_Appv'
]

FEATURES_SYNTH = FEATURES_REAL + [
    'CreditScore',
    'InterestRate',
    'DSCR',
    'NetIncomeBeforeDebt'
]

FEATURE_DESCRIPTIONS = {
    'Term': 'Loan term in months',
    'NoEmp': 'Number of employees at application',
    'NewExist': 'Existing (0) vs. new (1) business',
    'CreateJob': 'Number of jobs created',
    'RetainedJob': 'Number of jobs retained',
    'FranchiseCode': 'Franchise code identifier',
    'UrbanRural': 'Urban (1) vs. rural (0) location',
    'RevLineCr': 'Revolving credit line flag',
    'LowDoc': 'Low documentation loan flag',
    'DisbursementGross': 'Gross disbursement amount',
    'BalanceGross': 'Outstanding loan balance',
    'GrAppv': 'Gross approval amount',
    'SBA_Appv': 'SBA approved amount',
    'CreditScore': 'Synthetic credit score (simulated)',
    'InterestRate': 'Synthetic interest rate (simulated)',
    'DSCR': 'Debt service coverage ratio',
    'NetIncomeBeforeDebt': 'Synthetic net income before debt'
}


# ─── DATA LOADING & PREPROCESSING ────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Filter status and create target
    df = df[df['MIS_Status'].isin(['P I F', 'CHGOFF'])].copy()
    df['Defaulted'] = df['MIS_Status'].map({'P I F': 0, 'CHGOFF': 1})

    # Clean monetary columns
    for col in ['DisbursementGross', 'GrAppv', 'SBA_Appv', 'BalanceGross']:
        if col in df.columns:
            df[col] = (
                df[col]
                .replace(r"[\$,]", "", regex=True)
                .pipe(pd.to_numeric, errors='coerce')
            )

    # Convert numeric fields
    for col in ['Term', 'NoEmp', 'NewExist', 'CreateJob', 'RetainedJob', 'FranchiseCode']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Binary flags
    if 'RevLineCr' in df.columns:
        df['RevLineCr'] = df['RevLineCr'].fillna('N').map({'Y': 1, 'N': 0})
    if 'LowDoc' in df.columns:
        df['LowDoc'] = df['LowDoc'].fillna('N').map({'Y': 1, 'N': 0})
    if 'NewExist' in df.columns:
        df['NewExist'] = df['NewExist'].map({'Existing': 0, 'New': 1}).fillna(0).astype(int)

    # Synthetic features
    np.random.seed(42)
    df['CreditScore'] = np.random.normal(670, 50, len(df)).astype(int).clip(500, 800)
    df['InterestRate'] = np.random.normal(0.08, 0.015, len(df)).clip(0.03, 0.15)
    df['NetIncomeBeforeDebt'] = np.random.normal(100000, 25000, len(df)).clip(20000, 200000)

    # Compute DSCR
    ir = df['InterestRate']
    tm = df['Term']
    df['AnnualDebtService'] = (
        df['DisbursementGross'] * ir /
        (1 - (1 + ir) ** (-tm / 12))
    )
    df['AnnualDebtService'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['AnnualDebtService'].fillna(df['AnnualDebtService'].median(), inplace=True)
    df['DSCR'] = df['NetIncomeBeforeDebt'] / df['AnnualDebtService']
    df['DSCR'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['DSCR'].fillna(df['DSCR'].median(), inplace=True)

    # Final fill
    for col in FEATURES_SYNTH:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    return df


# ─── MODEL LOADER ──────────────────────────────────────────────
@st.cache_resource
def load_models():             # ← no parameters
    model_real  = joblib.load(MODEL_REAL_PATH)
    model_synth = joblib.load(MODEL_SYNTH_PATH)
    return model_real, model_synth

# ─── OCR HELPERS ───────────────────────────────────────────────────────────



# ── EDIT THIS STRING to wherever your IT-installed tesseract.exe lives ──
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# -----------------------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD



def pdf_to_images(path: str, dpi: int = 300):
    """Render each PDF page to a PIL.Image via PyMuPDF (pure‑Python)."""
    doc   = fitz.open(path)
    zoom  = dpi / 72           # 72 dpi = base PDF resolution
    mat   = fitz.Matrix(zoom, zoom)
    pages = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.fromarray(
            np.frombuffer(pix.samples, dtype=np.uint8)
              .reshape(pix.height, pix.width, pix.n)
        )
        pages.append(img)
    return pages


def ocr_pdf(path: str) -> str:
    """OCR a PDF using PyMuPDF for rasterisation + Tesseract for text."""
    try:
        pages = pdf_to_images(path, dpi=300)
    except Exception as err:
        st.error(f"❗ PDF rasterisation failed — {err}")
        return ""

    try:
        return "\n".join(pytesseract.image_to_string(p) for p in pages)
    except pytesseract.pytesseract.TesseractNotFoundError:
        st.error(
            "❗ **Tesseract executable not found.**\n"
            "Install from https://github.com/UB-Mannheim/tesseract/wiki "
            "and update `TESSERACT_CMD` in the script."
        )
        return ""


def ocr_image(img: Image.Image) -> str:
    """OCR a single image file."""
    try:
        return pytesseract.image_to_string(img)
    except pytesseract.pytesseract.TesseractNotFoundError:
        st.error(
            "❗ Tesseract executable not found. "
            "Update `TESSERACT_CMD` to point to tesseract.exe."
        )
        return ""

# Updated MONEY_RX: handles commas, no commas, decimals, optional $
MONEY_RX = r"\$?((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"

def money(label, text):
    # Step 1: Try exact label-based search
    pattern = fr"{label}.*?{MONEY_RX}"
    m = re.search(pattern, text, re.I)
    if m and m.group(1):
        return float(m.group(1).replace(',', ''))

    # Step 2: Try fuzzy matching (any line with a similar label and a $-like number)
    fuzzy_lines = [
        line for line in text.splitlines()
        if label.lower() in line.lower() and re.search(MONEY_RX, line)
    ]
    for line in fuzzy_lines:
        m = re.search(MONEY_RX, line)
        if m and m.group(1):
            return float(m.group(1).replace(',', ''))

    # Step 3: As last resort, grab the first money value in the text
    m = re.search(MONEY_RX, text)
    if m and m.group(1):
        return float(m.group(1).replace(',', ''))

    return None


def parse_to_features(text):
    feats = {k:None for k in FEATURES_SYNTH}
    feats['DisbursementGross'] = money("disbursement",text)
    feats['SBA_Appv']          = money("sba\\s+approved",text)
    feats['GrAppv']            = money("gross\\s+approval",text)
    feats['BalanceGross']      = money("outstanding|balance",text)
    m = re.search(r"term.*?(\\d{1,3})\\s*months",text,re.I)
    if m: feats['Term']=int(m.group(1))
    return feats

# ─── UTILITY: PSI CALCULATION ─────────────────────────────────────────────────
def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    breaks = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    exp_perc = np.histogram(expected, bins=breaks)[0] / len(expected)
    act_perc = np.histogram(actual, bins=breaks)[0] / len(actual)
    exp_perc[exp_perc == 0] = 1e-6
    act_perc[act_perc == 0] = 1e-6
    return np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc))


# ─── DASHBOARD SECTIONS ───────────────────────────────────────────────────────
def show_statistics(df: pd.DataFrame):
    st.header("Descriptive Statistics")
    st.subheader("Default Rate")
    st.bar_chart(df['Defaulted'].value_counts(normalize=True))

    st.subheader("Feature Summary (Real)")
    st.write(df[FEATURES_REAL].describe().T)

    st.subheader("Correlation Matrix")
    corr = df[FEATURES_REAL + ['Defaulted']].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)


def show_eda(df: pd.DataFrame):
    st.header("Interactive EDA")
    choice = st.radio("Plot type:", ['Numeric', 'Categorical'])
    if choice == 'Numeric':
        col = st.selectbox("Numeric feature", FEATURES_REAL[:5])
        fig, ax = plt.subplots()
        sns.histplot(data=df, x=col, hue='Defaulted', bins=30, kde=True, ax=ax)
        ax.set_xlim(df[col].min(), df[col].max())
        st.pyplot(fig)
    else:
        col = st.selectbox("Categorical feature", ['UrbanRural', 'RevLineCr', 'LowDoc', 'NewExist'])
        chart = pd.crosstab(df[col], df['Defaulted'], normalize='index')
        st.bar_chart(chart)


def show_predict(model_real, model_synth, df: pd.DataFrame):
    st.header("Predict Default Risk")
    with st.form("predict_form"):
        inputs = {}
        for feature in FEATURES_SYNTH:
            default = float(df[feature].median())
            inputs[feature] = st.number_input(feature, value=default)
        submitted = st.form_submit_button("Predict")

    if submitted:
        # Build input DataFrame
        X_input = pd.DataFrame([inputs], columns=FEATURES_SYNTH)
        # Real-only model on real features
        X_real = X_input[FEATURES_REAL]
        prob_real = model_real.predict_proba(X_real)[:, 1][0]
        # Real+Synth model on all features
        prob_synth = model_synth.predict_proba(X_input)[:, 1][0]

        st.metric("Default Prob (Real-only)", f"{prob_real:.2%}")
        st.metric("Default Prob (Real+Synth)", f"{prob_synth:.2%}")

def show_comparison(model_real, model_synth, df: pd.DataFrame):
# show_comparison follows (unchanged signature)(model_real, model_synth, scaler, df: pd.DataFrame):
    st.header("Model Comparison")
    X_all = df[FEATURES_SYNTH]
    y = df['Defaulted']
    X_train, X_test, y_train, y_test = train_test_split(X_all, y, stratify=y, random_state=42)
    X_real_test = X_test[FEATURES_REAL]
    X_synth_test = X_test[FEATURES_SYNTH]

    # Accuracies
    acc_real = model_real.score(X_real_test, y_test)
    acc_synth = model_synth.score(X_synth_test, y_test)
    st.metric("Test Accuracy (Real-only)", f"{acc_real:.2%}")
    st.metric("Test Accuracy (Real+Synth)", f"{acc_synth:.2%}")

    # Confusion matrices
    cm_real = confusion_matrix(y_test, model_real.predict(X_real_test))
    cm_synth = confusion_matrix(y_test, model_synth.predict(X_synth_test))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(cm_real, annot=True, fmt='d', ax=axes[0]).set_title('Real-only')
    sns.heatmap(cm_synth, annot=True, fmt='d', ax=axes[1]).set_title('Real+Synth')
    st.pyplot(fig)

    # ROC Curves
    prob_real = model_real.predict_proba(X_real_test)[:, 1]
    prob_synth = model_synth.predict_proba(X_synth_test)[:, 1]
    fpr_r, tpr_r, _ = roc_curve(y_test, prob_real)
    fpr_s, tpr_s, _ = roc_curve(y_test, prob_synth)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr_r, tpr_r, label=f'Real-only AUC={roc_auc_score(y_test, prob_real):.3f}')
    ax2.plot(fpr_s, tpr_s, label=f'Real+Synth AUC={roc_auc_score(y_test, prob_synth):.3f}')
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.legend()
    st.pyplot(fig2)

    # Feature importances
    st.subheader("Feature Importances")
    fi_real = pd.Series(model_real.feature_importances_, index=FEATURES_REAL).sort_values(ascending=False)
    fi_synth = pd.Series(model_synth.feature_importances_, index=FEATURES_SYNTH).sort_values(ascending=False)
    st.bar_chart(fi_real)
    st.bar_chart(fi_synth)

def show_model_health(df: pd.DataFrame, model_synth):
    """Visualises calibration and data drift for the production model.
    * Calibration curve + Brier score
    * PSI drift on DSCR (first half vs second half of dataset)
    """
    st.header("Model Health & Drift")

    X = df[FEATURES_SYNTH]
    y = df['Defaulted']
    _, X_test, _, y_test = train_test_split(X, y, stratify=y, random_state=42)

    # --- Calibration -----------------------------------------------------
    proba = model_synth.predict_proba(X_test)[:, 1]
    p_true, p_pred = calibration_curve(y_test, proba, n_bins=10)
    fig, ax = plt.subplots()
    ax.plot(p_pred, p_true, 'o', label='Model')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Empirical default rate')
    ax.set_title('Calibration Curve')
    ax.legend()
    st.pyplot(fig)
    st.write("Brier Score:", brier_score_loss(y_test, proba))

    # --- PSI on DSCR ------------------------------------------------------
    mid = len(df) // 2
    psi_val = compute_psi(df['DSCR'].iloc[:mid].values,
                          df['DSCR'].iloc[mid:].values)
    st.metric("DSCR PSI (First vs Second Half)", f"{psi_val:.3f}")

# --- SHAP TAB (fixed interaction collapse) -------------------------------
def show_shap_explainability(df: pd.DataFrame, model_real, model_synth):
    """Render bar + beeswarm plots with **2‑D** SHAP matrices.
    Uses shap.Explainer, which always returns (n_samples, n_features)
    so the feature matrix and SHAP matrix shapes match.
    """
    st.header("Global SHAP Explainability")
    shap.initjs()

    # --- Real‑only -------------------------------------------------------
    st.subheader("Real‑only Model SHAP")
    samp_r = df[FEATURES_REAL].sample(n=min(200, len(df)), random_state=42)
    expl_r = shap.Explainer(model_real, samp_r)
    vals_r = expl_r(samp_r).values                # 2‑D array guaranteed
    shap.summary_plot(vals_r, samp_r, plot_type="bar", show=False)
    st.pyplot(plt.gcf()); plt.close()
    shap.summary_plot(vals_r, samp_r, show=False)
    st.pyplot(plt.gcf()); plt.close()

    # --- Real + Synthetic ----------------------------------------------
    st.subheader("Real+Synthetic Model SHAP")
    samp_s = df[FEATURES_SYNTH].sample(n=min(200, len(df)), random_state=42)
    expl_s = shap.Explainer(model_synth, samp_s)
    vals_s = expl_s(samp_s).values
    shap.summary_plot(vals_s, samp_s, plot_type="bar", show=False)
    st.pyplot(plt.gcf()); plt.close()
    shap.summary_plot(vals_s, samp_s, show=False)
    st.pyplot(plt.gcf()); plt.close()

   
       
def show_feature_info():
# show_feature_info follows():
    st.header("Feature Metadata")
    df_info = pd.DataFrame.from_dict(FEATURE_DESCRIPTIONS, orient='index', columns=['Description'])
    st.table(df_info)

def generate_fake_pdf() -> bytes:
    """Return an in‑memory SBA‑style PDF (no temp file left open).
    Uses a BytesIO buffer to avoid Windows file‑lock issues."""
    fake = Faker()
    from io import BytesIO
    buffer = BytesIO()

    c = canvas.Canvas(buffer, pagesize=LETTER)
    c.setFont("Helvetica", 12)

    # Header
    c.drawString(50, 740, f"Business Name: {fake.company()}")
    c.drawString(50, 720, f"Owner: {fake.name()}")
    c.drawString(50, 700, f"Address: {fake.address().replace(chr(10), ' ')}")

    # Numeric fields
    import random
    gross_disb = random.randint(50_000, 1_000_000)
    sba_appv   = int(gross_disb * random.uniform(0.5, 0.9))
    balance    = int(gross_disb * random.uniform(0.2, 0.8))
    term       = random.choice([60, 120, 180, 240, 300])

    c.drawString(50, 660, f"Disbursement Amount: ${gross_disb:,.0f}")
    c.drawString(50, 640, f"SBA Approved Amount: ${sba_appv:,.0f}")
    c.drawString(50, 620, f"Outstanding Balance: ${balance:,.0f}")
    c.drawString(50, 600, f"Term: {term} months")

    c.showPage(); c.save()
    buffer.seek(0)
    return buffer.read()


def show_sample_doc_tab():
    st.header("📝 Generate a Fake Loan Document")
    st.write("Use this synthetic PDF to test the Upload‑Doc tab without exposing real client data.")
    if st.button("Create sample PDF"):
        pdf_bytes = generate_fake_pdf()
        st.download_button(
            label="Download fake_statement.pdf",
            data=pdf_bytes,
            file_name="fake_statement.pdf",
            mime="application/pdf"
        )


# ─── NEW: DOCUMENT UPLOAD TAB ──────────────────────────────────────────────
def show_doc_upload(model_synth, df_template):
    st.header("📄  Upload Financial Document")
    up = st.file_uploader("PDF, PNG or JPG",type=['pdf','png','jpg','jpeg'])
    if not up: return
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, up.name)
        with open(path,'wb') as f: f.write(up.getvalue())
        text = ocr_pdf(path) if path.lower().endswith('.pdf') else ocr_image(Image.open(path))
    st.expander("OCR preview").write(text[:400] + (" …" if len(text)>400 else ""))

    feats = parse_to_features(text)
    st.subheader("Parsed features (editable):")
    df_in = pd.DataFrame([feats])
    edited = st.data_editor(df_in, num_rows="fixed")

    if st.button("Predict default risk"):
        # Merge with template medians for missing fields
        for col in edited.columns:
            if pd.isna(edited.loc[0,col]):
                edited.loc[0,col] = df_template[col].median()
        prob = model_synth.predict_proba(edited[FEATURES_SYNTH])[:,1][0]
        st.metric("Predicted Default Probability", f"{prob:.2%}")

# ─── MAIN ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    st.sidebar.title("Navigation")
    tabs = ["Sample Doc","Upload Doc","Statistics","EDA","Predict","Compare",
            "Health","Feature Explanation","Info"]
    tab = st.sidebar.radio("Go to", tabs)

    df = load_data(DATA_PATH)
    model_real, model_synth = load_models()

    if tab == "Sample Doc":
        show_sample_doc_tab()
    elif tab == "Upload Doc":
        show_doc_upload(model_synth, df)
    elif tab == "Statistics":
        show_statistics(df)
    elif tab == "EDA":
        show_eda(df)
    elif tab == "Predict":
        show_predict(model_real, model_synth, df)
    elif tab == "Compare":
        show_comparison(model_real, model_synth, df)
    elif tab == "Health":
        show_model_health(df, model_synth)
    elif tab == "Feature Explanation":
        show_shap_explainability(df, model_real, model_synth)
    else:
        show_feature_info()