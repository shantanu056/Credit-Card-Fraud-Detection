# Credit-Card-Fraud-Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-yellow)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

##  Overview

Credit card fraud represents a significant challenge in the financial industry. With the increasing reliance on online transactions, fraudsters exploit vulnerabilities, causing financial and reputational losses.

This project aims to **analyze transaction patterns**, **handle imbalanced data**, and **apply machine learning algorithms** to accurately detect fraudulent transactions.A key enhancement in this iteration is the integration of MLFlow for comprehensive tracking, comparison, and management of all trained models.

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

To handle the dataset's challenges, the following steps were taken:

**Handling Missing Values:** Imputing or dropping columns with high rates of missing data.

**Categorical Feature Encoding:** Converting nominal and ordinal features into numerical representations (e.g., One-Hot Encoding).

**Feature Scaling:** Standardizing or normalizing numerical features to ensure all algorithms treat them equally.

**Imbalance Handling:** Applying techniques like SMOTE (Synthetic Minority Over-sampling Technique) to balance the class distribution, which is crucial for training effective fraud detection models.

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

## 🔍 Model Evaluation & MLFlow Tracking

Multiple classification models were tested and tracked using MLFlow.

### 1️⃣ MLFlow Run Overview

Each model (e.g., Logistic Regression, Decision Tree, XGBoost) is logged as a separate run under a single MLFlow experiment, capturing metrics and artifacts.

### 2️⃣ Model Comparison

MLFlow's UI enables side-by-side comparison of:
- Accuracy
- Precision
- Recall
- F1-Score

### 3️⃣ Run Logging

Each run logs:
- Hyperparameters
- Evaluation metrics
- Trained model artifact (`.pkl`)

---

## 🧪 Models & Artifacts

* Sample Logged variables include-

| Model              | Logged Metrics     | Key Parameters             | Artifacts     |
|-------------------|--------------------|----------------------------|---------------|
| Logistic Regression | Accuracy, F1, Precision, Recall       | `C`, `solver`              | `model.pkl`   |
| Decision Tree       | Accuracy, F1, Precision, Recall        | `max_depth`, `min_samples_split` | `model.pkl`   |
| XGBoost             | Accuracy, F1, Precision, Recall        | `n_estimators`, `max_depth`, `gamma` | `model.pkl`   |

<img width="1919" height="955" alt="Credit Card Fraud Detection Mlflow 1" src="https://github.com/user-attachments/assets/705cddc9-5428-4d5c-b258-dd2e38199125" />

<img width="1533" height="987" alt="Credit Card Fraud Detection Mlflow 2" src="https://github.com/user-attachments/assets/f6c7dcb3-e3df-4f7f-a802-b73688c0f47a" />

<img width="1917" height="1001" alt="Credit Card Fraud Detection Mlflow 3" src="https://github.com/user-attachments/assets/b1cb3be4-c175-4a30-ab1c-0a7835eedb6e" />

---

##  How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/shantanu056/Credit-Card-Fraud-Detection.git
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
   * **Algorithms-MLFlow** → trains, evaluates and logs models 

5. Start MLFlow UI
bash
mlflow ui
Then open http://localhost:5000 in your browser to view tracked runs.

---

##  Tech Stack

* Python 
* Pandas, NumPy
* Matplotlib, Seaborn (EDA & Visualization)
* Scikit-learn
* XGBoost
* MLFlow (Model Tracking & Management)

---

##  Future Work

* Hyperparameter tuning with GridSearchCV/RandomizedSearchCV
* MLFlow Model Registry for version control
* Deploying model with Flask/Streamlit
* Real-time fraud detection API

---
 Author

 Shantanu Bharati

GitHub: shantanu056
