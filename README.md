# 🧠 Insurance Claim Prediction Tool

This project is a **Machine Learning-based prediction system** that forecasts which customers are likely to apply for an insurance claim in the future. By leveraging **XGBoost** and **Deep Neural Networks (DNN)**, the tool provides insights that can help insurance companies improve their **risk assessment**, **customer profiling**, and **strategic decision-making**.

---

## 🚀 Project Overview

Insurance companies often face challenges in identifying customers who are more likely to file claims in the near future. Predicting these customers in advance enables better **resource allocation**, **fraud prevention**, and **customer retention strategies**.

This project uses a combination of **XGBoost** and **Deep Learning models** to achieve high prediction accuracy. The trained models are exported as JSON files for easy integration into production environments.

---

## 🧩 Key Features

* Predicts potential claimants based on historical data
* Implements **data preprocessing**, **feature engineering**, and **model optimization**
* Uses **XGBoost** and **Deep Neural Networks (DNN)** for prediction
* Exports trained model as a **JSON file** for reuse
* Scalable and adaptable to various insurance datasets

---

## 🧱 Project Workflow

### 1. **Data Collection**

* Collect historical insurance data containing customer demographics, policy details, and claim history.

### 2. **Data Cleaning**

* Handle missing values using mean/median imputation or removal.
* Remove duplicate records.
* Standardize inconsistent data formats.
* Detect and handle outliers.

### 3. **Data Transformation**

* Encode categorical variables using **Label Encoding** or **One-Hot Encoding**.
* Normalize numerical features using **MinMaxScaler** or **StandardScaler**.
* Create new features such as “claim frequency” or “years with company” to improve model performance.
* Split the dataset into **training** and **testing** sets.

### 4. **Model Training**

* **XGBoost Model:**

  * Gradient boosting model optimized for tabular data.
  * Fine-tuned using hyperparameters such as learning rate, depth, and estimators.
* **Deep Neural Network (DNN):**

  * Multi-layer perceptron architecture.
  * Trained with **ReLU activation**, **Adam optimizer**, and **binary cross-entropy loss**.
  * Early stopping used to prevent overfitting.

### 5. **Model Evaluation**

* Evaluate using metrics such as:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * ROC-AUC curve



## 🛠️ Technologies Used

| Category         | Tools / Libraries          |
| ---------------- | -------------------------- |
| Language         | Python                     |
| Data Processing  | Pandas, NumPy              |
| Visualization    | Matplotlib, Seaborn        |
| Machine Learning | XGBoost, Scikit-learn      |
| Deep Learning    | TensorFlow / Keras         |
| Model Saving     | JSON                       |
| Environment      | Jupyter Notebook / VS Code |


---

## 👨‍💻 Author

**Vipul Durgade**
*Data Analyst | Machine Learning Enthusiast*
📧 Vipdurgade@gmail.com  

