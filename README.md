# 📌 SBA Loan Default Prediction

Predicting whether an SBA loan will default using machine learning with both real-world and synthetic data.

---

## 📝 Table of Contents

1. [About the Project](#about-the-project)
2. [Motivation](#motivation)
3. [Features](#features)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Visuals](#visuals)
8. [Contributing](#contributing)
9. [Credits](#credits)
10. [License](#license)

---

## 📖 About the Project

This project applies a Random Forest classifier to predict loan default for SBA loans using historical data. It features:

* A real data model trained on SBA public loan data
* A synthetic data model trained on the same data augmented with estimated financial features
* A Streamlit dashboard for visualizing results and exploring model performance

---

## 🚀 Motivation

This project represents my growth from beginner to builder. It combines my interests in machine learning, finance, and product development into something practical and valuable. I wanted to create a solution that could help analysts and lenders better understand borrower risk, while also continuing to improve my own technical and communication skills in the process.

Whether you're a researcher, an aspiring quant, or just curious about data-driven decision making, I hope this project offers insight and value.

---

## 🌟 Features

Dynamically adjust individual loan features (e.g., credit score, interest rate, number of employees) within the dashboard and receive updated default probability predictions in real time

Simulate realistic loan applications and financial statements by generating synthetic 'fake' documents for testing

Integrate an OCR (Optical Character Recognition) pipeline to extract structured data from uploaded documents and images, feeding that data into the model to estimate default probability

Clean and transform raw SBA loan data using robust preprocessing techniques

Engineer synthetic borrower features such as Credit Score, DSCR, Interest Rate (non-public estimates)

Train and evaluate machine learning models using classification metrics (accuracy, precision, recall, F1-score)

Compare model performance across real vs. synthetic data feature sets

Generate interpretable model insights using feature importance rankings

Explore predictions and performance in an interactive Streamlit dashboard (no image dependencies)

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sba-loan-predictor.git
   cd sba-loan-predictor
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # On Windows
   source .venv/bin/activate 
   ```

3. Manually install required packages:

   ```bash
   pip install streamlit pandas numpy scikit-learn matplotlib seaborn PyMuPDF pillow joblib openpyxl shap
   ```

4. Train the Model
    ```bash
   python src/train_model.py
    ```
6. Run the dashboard:

   ```bash
   streamlit run app.py
   ```

---

##  Usage

Once launched, the dashboard allows users to:

Toggle between models (real vs. synthetic features)

View feature importances with interactive visualizations

Compare evaluation metrics such as accuracy, precision, recall, F1-score, ROC AUC, and calibration curves

Upload loan application documents or images for OCR extraction and default probability prediction

Adjust any loan feature (e.g., interest rate, DSCR, number of employees) in real time to see updated default probabilities

Explore and interpret the drivers behind SBA loan defaults through clear, data-driven insights

---

## 🗂 Project Structure

```
LoanDefaultPredictionModel/
├── .venv/                  # Virtual environment
├── dashboard.py            # Streamlit dashboard application
├── Train_model.py          # Model training script
├── .gitignore              # Git ignore rules
├── LICENSE                 # License information
├── README.md               # Project documentation
└── pyvenv.cfg               # Virtual environment configuration
```

---

## 🖼 Visuals

![Dashboard Screenshot](https://via.placeholder.com/800x400.png?text=Insert+your+dashboard+image+here)

> Dashboard showing model comparison and feature importances

---

## 🤝 Contributing

Interested in improving this project?

1. Fork it 🍴
2. Create a branch (`git checkout -b feature-idea`)
3. Commit changes (`git commit -m 'Add awesome feature'`)
4. Push your branch (`git push origin feature-idea`)
5. Open a Pull Request 🚀

Please ensure your PRs are well-tested and documented.

---

## 📊 Model Performance
After preprocessing and training on the SBA loan dataset (897,167 samples), the models produced the following results:

Real Features Model (13 columns):

Train Accuracy: 0.9961

Test Accuracy: 0.9294

AUC-ROC Score: 0.960

Brier Score: (not measured in this run)

False Positives: 5,952

False Negatives: 9,887

Real + Synthetic Features Model (17 columns):

Train Accuracy: 1.0000

Test Accuracy: 0.9324

AUC-ROC Score: 0.963

Brier Score: 0.05195

False Positives: 5,463

False Negatives: 9,699

Saved Models:

loan_model_real.pkl — Trained with real features only

loan_model_synth.pkl — Trained with both real and synthetic features

Performance Insights:

Synthetic features provided a modest improvement in both test accuracy and AUC-ROC.

The Real+Synthetic model maintained strong calibration, as indicated by the low Brier score.

Both models achieved high AUC-ROC values, indicating excellent discriminatory power.

Lower false positives and false negatives in the Real+Synthetic model suggest better generalization.
---

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

---

## 🏷️ Badges

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-yellow)
![Streamlit](https://img.shields.io/badge/WebApp-Streamlit-red)

---

## ✅ Tests

Basic model validation metrics are printed to the console during training:

* Accuracy
* Precision
* Recall
* F1-score

You can extend this with unit tests using `pytest` or `unittest` in the future.

---

## 🧠 Final Thoughts

> "A good README sets the tone for a great project."

Thank you for checking out this project. Feel free to fork, star, or reach out with ideas.

---

**Connect with me**
GitHub: [your-username](https://github.com/your-username)
Twitter: [@yourhandle](https://twitter.com/yourhandle)
LinkedIn: [Your Name](https://linkedin.com/in/yourname)

Happy Coding! ❤️
