# 🏠 House Price Prediction – Chennai

*A Machine Learning Model and Streamlit Web Application*

---

## 📘 **Abstract**

As the housing market in Chennai continues to expand, the need for reliable and intelligent price prediction systems has become increasingly essential for homebuyers, sellers, and real estate stakeholders. House prices are influenced by numerous unpredictable factors, making estimation difficult without data-driven support.

This project presents a comprehensive **House Price Prediction model** that leverages machine learning regression techniques and advanced feature selection methods to provide accurate price forecasts. The dataset includes detailed information such as **location**, **square footage**, **number of bedrooms**, **bathrooms**, **parking**, **zone**, and **proximity to main roads**.

To identify the most influential variables, multiple feature-selection techniques were applied, including:

* Feature Correlation
* Mutual Information Regression
* LASSO Regression
* Boruta
* Sequential Feature Selection

For the predictive model, **Random Forest Regression** demonstrated the most robust performance and was deployed as the final model.

This end-to-end approach aims to deliver a reliable tool that simplifies real-estate decisions in Chennai’s dynamic housing market.

---

## 🎯 **Project Goals**

* Accurately predict house prices using ML regression techniques
* Identify significant real-estate features through rigorous feature selection
* Build a clean and interactive **Streamlit web app** for users
* Provide insights into factors affecting housing prices in Chennai

---

## 🧠 **Machine Learning Approach**

### **1. Data Preprocessing**

* Handling missing values
* Encoding categorical variables
* Normalization / Standardization
* Outlier analysis

### **2. Feature Selection Techniques**

* Pearson Correlation
* Mutual Information Regression
* LASSO Regression
* Boruta Algorithm
* Sequential Feature Selector

### **3. Models Trained**

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor

### **Final Model Used:**

✔ **Random Forest Regressor**

---

## 🖥️ **Streamlit Web App Features**

* Simple and interactive UI
* Dropdown for area (Adyar, Anna Nagar, Chrompet, KK Nagar, Karapakkam, T Nagar)
* Inputs for square footage, bedrooms, bathrooms, parking, zone, registration, commission
* Auto-calculation of **Price per Square Foot**
* Model auto-extracted from ZIP during deployment
* Works on both local machine & Streamlit Cloud

---

## 📦 **Tech Stack**

* **Python 3.x**
* **Streamlit**
* **NumPy**, **Pandas**
* **Scikit-Learn**
* **XGBoost**
* **Statsmodels**
* **Matplotlib**, **Seaborn**
* **Pickle**

---

## 🚀 **How to Run Locally**

### **1. Clone the repo**

```
git clone https://github.com/JaysreeSS/house-price-prediction.git
cd house-price-prediction
```

### **2. Install dependencies**

```
pip install -r requirements.txt
```

### **3. Run Streamlit**

```
streamlit run app.py
```
