# Credit-Card-Fraud-Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-yellow)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

##  Overview

Credit card fraud represents a significant challenge in the financial industry. With the increasing reliance on online transactions, fraudsters exploit vulnerabilities, causing financial and reputational losses.

This project aims to **analyze transaction patterns**, **handle imbalanced data**, and **apply machine learning algorithms** to accurately detect fraudulent transactions.

---

##  Project Structure

```
Credit-Card-Fraud-Detection/
│
├── Credit Card Fraud Detection (Preprocessing).ipynb   # Data cleaning, EDA, preprocessing
├── Credit Card Fraud Detection (Algorithms).ipynb      # Model building & evaluation
├── README.md                                           # Project documentation
└── requirements.txt                                    # Dependencies
```

---

##  Dataset

* **Size**: 100,000 transactions
* **Type**: Real-world credit card activity
* **Features**:

  * Independent variables: Transaction amount, TransactionID, TransactionDate, TransactionType, etc.
  * Dependent variable: IsFraud (0 = Legitimate, 1 = Fraud)

>  The dataset is highly **imbalanced**, with fraudulent transactions making up a very small percentage of total records.

---

##  Data Preprocessing

* Loaded dataset and explored columns, datatypes, and null values.
* Conducted **exploratory data analysis (EDA)** with statistical summaries and visualizations.
* Addressed **class imbalance** using resampling techniques.
* Scaled/normalized features where necessary.
* Finalized a **balanced, clean dataset** for modeling.

---

##  Machine Learning Models

The following algorithms were implemented and evaluated:

* ✅ Logistic Regression
* ✅ K-Nearest Neighbors (optimized with best K)
* ✅ Support Vector Machine (SVM with tuned parameters)
* ✅ Decision Tree
* ✅ Random Forest
* ✅ AdaBoost
* ✅ Gradient Boost
* ✅ XGBoost

---

##  Results & Findings

* All algorithms achieved **similar accuracy**.
* **Decision Tree** was chosen as the final model because it achieved **comparable accuracy with the least processing time**.

| Model               | Accuracy | Notes                           |
| ------------------- | -------- | ------------------------------- |
| Logistic Regression | \~ High  | Stable baseline model           |
| KNN                 | \~ High  | Tuned for best neighbors        |
| SVM                 | \~ High  | Required parameter tuning       |
| Decision Tree       | \~ High  |  Fastest, chosen final model   |
| Random Forest       | \~ High  | Slightly higher processing time |
| AdaBoost            | \~ High  | Boosting improved stability     |
| Gradient Boost      | \~ High  | Similar performance to others   |
| XGBoost             | \~ High  | Competitive but slower          |

---

##  How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/Credit-Card-Fraud-Detection.git
   cd Credit-Card-Fraud-Detection
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Open Jupyter Notebook:

   ```bash
   jupyter notebook
   ```
4. Run notebooks in order:

   * **Preprocessing** → prepares the data
   * **Algorithms** → trains & evaluates models

---

##  Tech Stack

* Python 
* Pandas, NumPy
* Matplotlib, Seaborn (EDA & Visualization)
* Scikit-learn
* XGBoost

---

##  Future Work

* Hyperparameter tuning with GridSearchCV/RandomizedSearchCV
* Deploying model with Flask/Streamlit
* Real-time fraud detection API

---
 Author

 Shantanu Bharati

GitHub: shantanu056
