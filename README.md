# From-Data-to-Dollars---Predicting-Insurance-Charges

## 📌 Overview
This project analyzes medical insurance data to predict patient **charges** using regression techniques.  
The main goal is to build a reliable model that meets the project requirement of achieving an **R² score above 0.65**.  

I experimented with different regression models and  used **Ridge Regression** with hyperparameter tuning with **GridSearchCV** for better performance.  

---

## 📂 Dataset
The dataset was sourced from multiple **sample data providers**:
- ❄️ Snowflake: London Public Transport, Soccer  
- 🟥 Redshift: London Public Transport  
- 🟦 Google BigQuery: London Public Transport  
- 🟪 Databricks: Projects Data  
- 🟧 PostgreSQL: Projects Data  

### Dataset Summary
- **Rows:** 1,338 (1,284 without N/A)  
- **Columns:** 17 (after preprocessing)  
- **Target Variable:** `charges`  

### Data Issues & Fixes
- `sex`: multiple formats like (`male`, `M`, `man`) unified to binary (`1 = male`, `0 = female`).  
- `region`: case-sensitive duplicates were standardized.  
- `charges`: some entries contained `$`; removed before converting to float.  
- Missing values: dropped for the target value `charges` to avoid bias and miscalculations, else Imputation was used through **SimpleImputer**.  

---

## 🔧 Preprocessing
- **Encoding:** One-Hot Encoding for categorical features (`region`, `smoker`).  
- **Manual Mapping:** For `sex` column.  
- **Scaling:** Only applied to feature set `X` (not the target).  

---

## 🤖 Modeling
Models considered:
- **Linear Regression**  
- **Ridge Regression** (chosen)  

Why Ridge?  
- Regularization helps prevent overfitting.  
- Suitable for datasets with a few number of features.  

### Hyperparameter Tuning
I used **GridSearchCV** with 5-fold cross-validation:  
```python
param_grid = {'alpha': np.linspace(0.01, 1, 20)}
model = GridSearchCV(Ridge(), param_grid=param_grid, cv=5, scoring='r2')
```
## 📊 Results
R² Score (Test Set): 0.65+ ✅ (requirement met)

Best `alpha`: Selected via GridSearchCV

## How to run
git clone https://github.com/MohamedElwan15/From-Data-to-Dollars---Predicting-Insurance-Charges
cd insurance-charges-prediction

## 🔮 Future Improvements
Try Lasso Regression for stronger feature selection.

Explore non-linear models (Random Forest, XGBoost).

Perform feature engineering (interaction terms, polynomial features).

## 👤 Author

Developed by Mohamed Elwan.

Part of a machine learning project for predictive modeling.
