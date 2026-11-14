# California Housing Price Prediction

An end-to-end machine learning project implementing a Linear Regression model to predict median house values in California districts using the classic California Housing dataset.

---

## 📊 Project Overview

This project demonstrates a complete data science and machine learning workflow in Python—from exploratory data analysis (EDA) and data preprocessing to model training, evaluation, and residual analysis[cite: 4]. 

---

## 🗂️ Dataset Characteristics

* **Source**: Derived from the 1990 U.S. Census via the StatLib repository, loadable via `scikit-learn` (`fetch_california_housing`)[cite: 4].
* **Number of Instances**: 20,640 block groups[cite: 4].
* **Target Variable**: `Price` (`MedHouseVal`) – Median house value for California districts, expressed in hundreds of thousands of dollars ($100,000)[cite: 4].
* **Features (8 Numeric Attributes)**[cite: 4]:
  * `MedInc`: Median income in block group[cite: 4]
  * `HouseAge`: Median house age in block group[cite: 4]
  * `AveRooms`: Average number of rooms per household[cite: 4]
  * `AveBedrms`: Average number of bedrooms per household[cite: 4]
  * `Population`: Block group population[cite: 4]
  * `AveOccup`: Average number of household members[cite: 4]
  * `Latitude`: Block group latitude[cite: 4]
  * `Longitude`: Block group longitude[cite: 4]

---

## 🛠️ Tech Stack & Libraries

* **Python**[cite: 4]
* **Pandas & NumPy** (Data manipulation and numerical operations)[cite: 4]
* **Matplotlib & Seaborn** (Data visualization, correlation heatmaps, and residual plots)[cite: 4]
* **Scikit-Learn** (Dataset loading, train-test splitting, data scaling, Linear Regression, and evaluation metrics)[cite: 4]

---

## 🚀 Workflow & Pipeline

1. **Data Loading & Inspection**: Fetched the dataset, checked data types, verified that there are no missing values (`df.isnull().sum()`), and generated statistical summaries (`df.describe()`)[cite: 4].
2. **Exploratory Data Analysis (EDA)**: Computed correlation matrices (`df.corr()`) and visualized feature correlations using a Seaborn heatmap
3. **Data Preprocessing & Splitting**:
   * Divided the dataset into features (`x`) and target (`y`)
   * Performed an 80-20 train-test split (`train_test_split` with `test_size=0.2` and `random_state=42`)
   * Standardized features using `StandardScaler`
4. **Model Training**: Trained a `LinearRegression` model on the training data
5. **Evaluation**: Assessed model performance on the test set using:
   * Mean Squared Error (MSE) / Root Mean Squared Error (RMSE)
   * Mean Absolute Error (MAE)
   * $R^2$ Score & Adjusted $R^2$ Score
6. **Diagnostic Visualization**: Visualized predictions vs. true test values using scatter plots and analyzed residuals

---

## ⚙️ Installation & Usage

To run this notebook locally, ensure you have Python installed along with the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
