# Hr_comma_sep-Turnover-Prediction-using-py
A project analyzing and predicting employee turnover using machine learning models.

# HR Employee Turnover Prediction

## Project Overview
This project aims to analyze and predict employee turnover in an organization. It uses machine learning models to identify key factors contributing to turnover and evaluate their predictive power. The analysis includes data visualizations and performance comparisons of different classifiers.

## Tools and Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset
The dataset contains the following attributes:
- satisfaction_level: Employee satisfaction level (0 to 1)
- last_evaluation: Last evaluation score (0 to 1)
- number_project: Number of projects completed by the employee
- average_monthly_hours: Average monthly working hours
- time_spend_company: Years spent in the company
- Work_accident: Whether the employee had an accident (binary)
- promotion_last_5years: Whether the employee was promoted in the last 5 years (binary)
- Department: Employee's department
- salary: Salary level (low, medium, high)

## Models Implemented
- Decision Tree
- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

## Key Results
| Model                | Accuracy | Recall  | Precision | F1-Score |
|----------------------|----------|---------|-----------|----------|
| Decision Tree        | 96.8%    | 91.4%   | 94.7%     | 93.0%    |
| K-Nearest Neighbors  | 96.2%    | 91.9%   | 91.9%     | 91.9%    |
| Random Forest        | 95.2%    | 81.2%   | 98.3%     | 88.9%    |
| Logistic Regression  | 87.1%    | 65.9%   | 75.6%     | 70.4%    |
| Support Vector Machine (SVM) | 87.4% | 76.3% | 71.7% | 73.9% |

## Visualizations
The project includes the following visualizations:
- Employee distribution by department
- Proportion of employees who stayed vs. left
- Salary distribution by overwork status
- ROC curve comparison for all models



