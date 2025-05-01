# churn_prediction.py

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import warnings

warnings.filterwarnings('ignore')

# Step 2: Load Dataset
df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\Customer churn prediction\Churn_Modelling.csv")

# Step 3: Basic EDA
print("First 5 rows:\n", df.head())
df.info()
print("\nSummary Statistics:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())

# Step 4: Drop Irrelevant Columns
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Step 5: Handle Missing Values
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('Exited')
categorical_cols = df.select_dtypes(include=['object']).columns

df[numerical_cols] = SimpleImputer(strategy='mean').fit_transform(df[numerical_cols])
df[categorical_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[categorical_cols])

# Step 6: Encode Categorical Variables
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Step 7: Split Features and Target
X = df.drop('Exited', axis=1)
y = df['Exited']

# Step 8: Apply SMOTE for Class Balancing
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X, y)

# Step 9: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 10: Train Model (XGBoost)
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Step 11: Evaluate Model
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 12: Feature Importance Visualization
feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns, columns=['Importance'])
top_features = feature_importances.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=top_features.Importance, y=top_features.index)
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.show()

# Step 13: Save Model and Feature Columns
joblib.dump(model, 'churn_model.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')
print("\n✅ Model and feature columns saved successfully!")
