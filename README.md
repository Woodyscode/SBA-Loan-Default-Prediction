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

* ✅ Data preprocessing pipeline
* ✅ Feature engineering (synthetic financial variables like Credit Score, DSCR)
* ✅ Train/test split with performance evaluation
* ✅ Comparison of real vs. synthetic feature models
* ✅ Visual feature importance ranking
* ✅ Interactive Streamlit dashboard

---

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/sba-loan-predictor.git
   cd sba-loan-predictor
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Train the models:

   ```bash
   python src/train_model.py
   ```

4. Run the dashboard:

   ```bash
   streamlit run app.py
   ```

---

## 🧠 Usage

Once launched, the dashboard allows users to:

* Toggle between models (real vs synthetic features)
* View feature importances
* Compare metrics like accuracy, precision, recall, and F1-score
* Understand the drivers of SBA loan defaults

---

## 🗂 Project Structure

```
sba-loan-predictor/
├── data/                  # Data files (real & augmented)
├── models/                # Saved model files (.pkl)
├── src/                   # Core code: training, preprocessing
├── app.py                 # Streamlit app
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
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

## 🙌 Credits

Built with guidance from various open-source examples, tutorials, and documentation including:

* [Scikit-learn Documentation](https://scikit-learn.org/stable/)
* [Streamlit Documentation](https://docs.streamlit.io/)
* Special thanks to my mentors and colleagues

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
