import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

#Load datasets
train_df = pd.read_csv("Dataset/UNSW_NB15_training-set.csv")
test_df = pd.read_csv("Dataset/UNSW_NB15_testing-set.csv")
#Drop ID column
train_df.drop(columns=['id'], inplace=True)
test_df.drop(columns=['id'], inplace=True)
#Separate features and labels
X_train = train_df.drop(columns=['label', 'attack_cat'])
y_train = train_df['label']
X_test = test_df.drop(columns=['label', 'attack_cat'])
y_test = test_df['label']
#Identify categorical and numerical columns
categorical_cols = ['proto', 'service', 'state']
numerical_cols = [col for col in X_train.columns if col not in categorical_cols]
#Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ]
)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
#Train baseline Random Forest model
rf_baseline = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_baseline.fit(X_train_processed, y_train)
#Predictions
y_pred = rf_baseline.predict(X_test_processed)
#Evaluation
print("Baseline Random Forest Results\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))