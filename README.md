# **RealtyNova - Predictive Analytics for Real Estate Valuation**

![c7dbfec0-e3f7-4948-8478-c9a4ee0f83fc](https://github.com/user-attachments/assets/f9d4c5d3-24df-4907-a96a-481e38b80299)

## **Objective**

The goal of this project is to **predict residential property prices** using machine learning techniques, while identifying the most influential features that determine property value. Additionally, the project demonstrates how to **visualize predictions and insights through an interactive dashboard**, making the results actionable for real estate analytics or decision-making.

By completing this project, we explore **end-to-end predictive analytics** workflows — from data cleaning and exploratory analysis to model building, evaluation, and visualization.

---

## **Dataset**

* **Source:** [Kaggle — Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
* **Number of Features:** 77 attributes describing residential properties
* **Key Features Include:**

  * **OverallQual:** Overall material and finish quality
  * **GrLivArea:** Above-grade living area (sq ft)
  * **GarageCars / GarageArea:** Garage capacity and size
  * **TotalBsmtSF:** Total basement area
  * **Neighborhood:** Geographical location within the city
  * **HouseStyle / BldgType:** Type and style of dwelling
* **Target Variable:** `SalePrice` — the selling price of the property in USD

This dataset offers a rich mix of **numerical and categorical variables**, enabling us to demonstrate practical feature engineering, handling of missing values, and predictive modeling.

---

## **Tools & Technologies**

* **Programming & Data Analysis:**

  * **Python** libraries: Pandas, NumPy, Matplotlib, Seaborn for data processing, exploration, and visualization
* **Machine Learning:**

  * **Linear Regression:** Simple and interpretable baseline model
  * **Random Forest:** Handles non-linear relationships and provides feature importance
  * **XGBoost:** Gradient boosting ensemble achieving highest predictive performance
* **Visualization & Dashboarding:**

  * **Power BI:** Interactive dashboards for exploring actual vs predicted prices, feature relationships, and neighborhood trends
* **Version Control & Collaboration:**

  * Git & GitHub for tracking code, scripts, and project updates

Absolutely! Here’s a **fully expanded, professional, recruiter-friendly version** of your Methodology, Results, Insights, and Next Steps sections. You can directly include this in your GitHub README:

---

## **Methodology**

### **1. Data Cleaning & Preprocessing**

* **Handling Missing Values:** Columns with excessive missing data (e.g., PoolQC, Alley, Fence) were dropped. Remaining missing values were imputed using medians for numerical features (e.g., LotFrontage by Neighborhood) and mode or placeholder strings for categorical features (e.g., GarageType, BsmtQual).
* **Feature Engineering:** Categorical features were encoded to allow machine learning models to interpret them. New features, such as total rooms, total basement area, and combined garage metrics, were derived to improve predictive performance.
* **Column Selection:** Non-informative or redundant columns like `Id` were removed to prevent noise in modeling.

---

### **2. Exploratory Data Analysis (EDA)**

* **Distribution Analysis:** Examined distributions of numerical variables like SalePrice, GrLivArea, LotArea, and OverallQual to detect skewness and outliers.
* **Correlation Analysis:** Identified features most correlated with SalePrice, including `OverallQual`, `GrLivArea`, `GarageCars`, `GarageArea`, and `TotalBsmtSF`.
* **Visual Insights:** Used scatter plots and boxplots to visualize the relationships between key features and property prices, revealing clear trends and patterns.

---

### **3. Model Training & Evaluation**

* **Models Used:**

  * **Linear Regression:** Baseline model to capture linear relationships.
  * **Random Forest Regressor:** Handles non-linear relationships and provides feature importance.
  * **XGBoost Regressor:** Gradient boosting ensemble method providing high accuracy and robustness.
* **Evaluation Metrics:**

  * **Root Mean Squared Error (RMSE):** Measures prediction error magnitude.
  * **R² Score:** Quantifies the proportion of variance explained by the model.
* **Results:**

| Model             | RMSE       | R² Score  |
| ----------------- | ---------- | --------- |
| Linear Regression | 35,339     | 0.837     |
| Random Forest     | 29,729     | 0.885     |
| **XGBoost**       | **28,563** | **0.894** |

**Best Model:** XGBoost was selected for its superior performance and ability to capture complex relationships in the data.

---

### **4. Dashboard & Visualization**

* **Actual vs Predicted Prices:** Scatter plots to validate model predictions visually.
* **Feature vs Price Correlations:** Visualized key drivers such as GrLivArea, OverallQual, and GarageCars.
* **Neighborhood & HouseStyle Trends:** Showed how location and style influence property values.
* **Interactive Insights:** Built in Power BI, enabling dynamic exploration of relationships between features and prices.

---

### **5. Prediction Export**

* Model predictions were saved in CSV format for further analysis and dashboard integration.
* Facilitates sharing insights with stakeholders or integrating into larger analytics workflows.

---

## **Key Insights**

* **Quality & Area Matter Most:** Higher `OverallQual` and `GrLivArea` consistently predict higher property prices.
* **Garage & Basement Features Influence Value:** Number of garage cars, garage area, and total basement area positively impact prices.
* **Neighborhood Effect:** Location is a critical factor; certain neighborhoods consistently fetch higher sale prices.
* **Non-Linear Relationships:** Ensemble models like Random Forest and XGBoost capture complex interactions better than linear models.

---

## **Takeaways**

* **Ensemble Methods Improve Accuracy:** Combining multiple decision trees captures complex patterns that linear models cannot.
* **Data Preprocessing is Key:** Proper handling of missing values, outliers, and feature engineering significantly boosts model performance.
* **Visualization Drives Insights:** Dashboards enable stakeholders to quickly understand predictive patterns and trends.
* **End-to-End Pipeline:** From cleaning to model deployment, a structured workflow ensures reproducibility and scalability.

---



