# Customer Churn Prediction

## 📌 Project Overview

This project aims to predict whether a customer will leave a bank (churn) based on various features such as credit score, age, tenure, balance, number of products, and more. It leverages a machine learning pipeline including data preprocessing, model training, evaluation, and a Streamlit web application for user-friendly prediction.

---

## 🗂️ Repository Structure


Customer-Churn-Prediction/
│
├── .gitignore                # Specifies intentionally untracked files to ignore
├── LICENSE                   # MIT License for open-source distribution
├── README.md                 # Project documentation
├── Churn_Modelling.csv       # Dataset used for training and testing
├── app.py                    # Streamlit web app for live predictions
├── churn_model.pkl           # Trained machine learning model (pickle format)
├── churn_prediction.py       # Core script for model loading and prediction logic
├── feature_columns.pkl       # Pickled list of feature columns used in model
├── requirement.txt           # Python dependencies (should be renamed to requirements.txt)
├── tempCodeRunnerFile.py     # Temporary file (can be deleted)


## ⚙️ Technologies Used

- Python 3.x
- Pandas & NumPy
- Scikit-learn
- Streamlit
- Pickle
- Matplotlib / Seaborn (for visualization, optional)

---

## 🧠 Model Workflow

1. **Load Data** – from `Churn_Modelling.csv`
2. **Preprocess Features** – Handle categorical variables, normalization
3. **Feature Engineering** – Selected relevant features and saved them as `feature_columns.pkl`
4. **Train Model** – Trained using Random Forest or similar classifier
5. **Model Evaluation** – Measured using Accuracy, Precision, Recall, F1-Score
6. **Model Deployment** – Deployed via Streamlit in `app.py`

---

## 🖥️ How to Run

### 1. Install Requirements


pip install -r requirements.txt


### 2. Run Streamlit App


streamlit run app.py


### 3. Model Prediction (Command Line)

You can also run predictions via:


python churn_prediction.py
-------

## 🌐 Live Demo

The Customer Churn Prediction app is live! 🚀  
You can test it here:

🔗 [Launch App](https://dudhesakshi-customer-churn-prediction-app-tnaemx.streamlit.app)



## 🧪 Sample Input Features

- Credit Score  
- Geography  
- Gender  
- Age  
- Tenure  
- Balance  
- Number of Products  
- Has Credit Card  
- Is Active Member  
- Estimated Salary

---

## 📈 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix (optional visualization)

---

## 🚀 Future Improvements

- Add hyperparameter tuning (GridSearchCV or Optuna)
- Enhance UI with input validation
- Add support for real-time data entry
- Deploy to cloud (e.g., Heroku, AWS, etc.)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---

## 🤝 Contributions

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📬 Contact

Maintainer: **Sakshi Dudhe** 
GitHub: [@Dudhesakshi](https://github.com/Dudhesakshi)
