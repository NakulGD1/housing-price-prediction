🏠 House Price Prediction (Regression)

A complete machine learning project built using Jupyter Notebook to predict house prices using regression techniques.
This project includes feature selection, multiple ML models, and RMSE comparison using the Housing.csv dataset.

📌 Project Overview
This project applies regression algorithms to predict housing prices based on several numerical and categorical features.
Key steps performed:
Data loading & cleaning
Label encoding categorical columns
Exploratory Data Analysis (EDA)
Correlation heatmap for feature selection
Training multiple regression models
Comparing performance using RMSE
Identifying the best-performing model

📊 Dataset
The dataset used is Housing.csv, containing:
Numerical features
Categorical features (converted to numeric using label encoding)
Target variable: price

🛠️ Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-Learn
Jupyter Notebook

🧹 Data Preprocessing
Removed missing values
Encoded categorical columns using label encoding
Selected strongest features using correlation analysis
Split into train/test sets (80/20)

🤖 Machine Learning Models Trained
Model	Description
Linear Regression	Baseline simple regression
Ridge Regression	Regularized linear model
Decision Tree Regressor	Non-linear tree model
Random Forest Regressor	Ensemble model with multiple trees
📈 Model Evaluation

Models were compared using Root Mean Squared Error (RMSE).
Example result format:
{
  'Linear Regression': 43000,
  'Ridge Regression': 42500,
  'Decision Tree': 52000,
  'Random Forest': 31000
}


The model with the lowest RMSE was selected as the best performer.
🗂️ Repository Structure
├── House Price Prediction.ipynb

├── Housing.csv

└── README.md

🚀 How to Run
Clone the repository:
git clone <your-repo-link>

Install required libraries:
pip install pandas numpy matplotlib seaborn scikit-learn

Open the notebook:
jupyter notebook

Run all cells to reproduce the results.

⭐ Future Improvements
Hyperparameter tuning (GridSearchCV)
Cross-validation
Model saving (Pickle)
Streamlit or Flask web app
Feature scaling & more advanced preprocessing

📬 Contact
If you want help improving the project or turning it into a portfolio-ready ML project, feel free to ask!
