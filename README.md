# Telco Customer Churn Prediction (End-to-End ML Deployment)
This project predicts customer churn using machine learning techniques applied to IBM’s Telco Customer Churn dataset. The objective is to identify high-risk customers early to enable proactive retention strategies. The solution covers the complete lifecycle from data preprocessing and model development to containerized cloud deployment.

## Dataset 📁
IBM Telco Customer Churn Dataset
1. [Old version:](https://www.kaggle.com/datasets/blastchar/telco-customer-churn?select=WA_Fn-UseC_-Telco-Customer-Churn.csv)
2. [New version](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)

## Software and Tools Requirements 🛠️
1. [Github Account](https://github.com)
2. [HerokuAccount](https://heroku.com)
3. [VSCodeIDE](https://code.visualstudio.com/)
4. [GitCLI](https://git-scm.com/book/en/v2/Getting-Started-The-Command-Line)
5. [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)

## Application Interface 📸
1. UI - Heroku & Docker ![Churn Web App UI Heroku & Docker](screenshots/ui_heroku.png)
2. Docker COntainer on Heroku ![Docker Container on Heroku](screenshots/docker_container.png)
3. Postman REST API Validation ![Validated Churn REST API with Postman ](screenshots/postman.png)

# Methodology 🧠
## 1. Data Preparation
- Data cleaning & missing value handling  
- Feature engineering  
- Categorical encoding  
- Outlier detection  
- Standardization  
- Multicollinearity detection (VIF)  
- Statistical testing (Chi-Square, Correlation analysis)  

## 2. Exploratory & Statistical Analysis
- Survival analysis  
- Spearman correlation heatmap  
- Normality testing (QQ plots)  
- Class imbalance assessment  

## 3. Model Development
Trained and evaluated:
- Random Forest  
- XGBoost  
- Artificial Neural Network  

## 4. Evaluation Metrics:
- Confusion Matrix  
- Classification Report  
- ROC-AUC Score  
- F1 Score  
- Recall (prioritized for churn detection)

## 5. Final Model Selection 🏆
👉 XGBoost selected based on superior Recall performance.
- Threshold tuning from 0.50 → 0.35 improved F1-score  
- Hyperparameter optimization using RandomizedSearchCV  
- SHAP used for interpretability and top 12 feature selection  
- Model exported in both Pickle and XGBoost JSON format  
- JSON used in production for deployment stability

## 6. Deployment Architecture 🚀
👉 The solution is deployed as a production-ready web application:
- Flask backend (REST API)
- HTML frontend for user input
- Docker containerization
- CI/CD via GitHub Actions
- Deployed on Heroku Cloud (Container Stack)

## 🌐 Live Application
https://telco-churn-predictions-008962de0ed2.herokuapp.com/