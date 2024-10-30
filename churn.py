import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv(r'C:\Users\DELL\Desktop\Customer Churn Prediction\Churn_Modelling.csv', encoding='latin-1')

# 1. Data Preprocessing
# Drop unnecessary columns
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Encode categorical variables
# Label encode 'Gender' column
data['Gender'] = LabelEncoder().fit_transform(data['Gender'])

# One-hot encode 'Geography' column
data = pd.get_dummies(data, columns=['Geography'], drop_first=True)

# Separate features and target variable
X = data.drop('Exited', axis=1)
y = data['Exited']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Model Building
# Logistic Regression
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train, y_train)
log_y_pred = log_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

# Gradient Boosting
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
gb_y_pred = gb_model.predict(X_test)

# 3. Evaluation
def evaluate_model(model_name, y_test, y_pred):
    print(f"\n{model_name} Model Performance")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Evaluate each model
evaluate_model("Logistic Regression", y_test, log_y_pred)
evaluate_model("Random Forest", y_test, rf_y_pred)
evaluate_model("Gradient Boosting", y_test, gb_y_pred)

# 4. Hyperparameter Tuning for Random Forest and Gradient Boosting
# Random Forest Hyperparameter Tuning
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=3, scoring='roc_auc')
rf_grid_search.fit(X_train, y_train)
print("\nBest Random Forest Parameters:", rf_grid_search.best_params_)

# Gradient Boosting Hyperparameter Tuning
gb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
gb_grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_param_grid, cv=3, scoring='roc_auc')
gb_grid_search.fit(X_train, y_train)
print("\nBest Gradient Boosting Parameters:", gb_grid_search.best_params_)

# Re-train the best Random Forest model with tuned parameters
best_rf_model = rf_grid_search.best_estimator_
best_rf_model.fit(X_train, y_train)
best_rf_y_pred = best_rf_model.predict(X_test)
evaluate_model("Tuned Random Forest", y_test, best_rf_y_pred)

# Re-train the best Gradient Boosting model with tuned parameters
best_gb_model = gb_grid_search.best_estimator_
best_gb_model.fit(X_train, y_train)
best_gb_y_pred = best_gb_model.predict(X_test)
evaluate_model("Tuned Gradient Boosting", y_test, best_gb_y_pred)
'''
# Optionally save the best model (for deployment)
import joblib
joblib.dump(best_gb_model, 'best_churn_model.pkl')
print("\nBest model saved as 'best_churn_model.pkl'")
'''