import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Load datasets
train_df = pd.read_csv("Dataset/UNSW_NB15_training-set.csv")
test_df = pd.read_csv("Dataset/UNSW_NB15_testing-set.csv")
# Drop ID column
train_df.drop(columns=['id'], inplace=True)
test_df.drop(columns=['id'], inplace=True)
# Separate features and labels
X_train = train_df.drop(columns=['label', 'attack_cat'])
y_train = train_df['label']
X_test = test_df.drop(columns=['label', 'attack_cat'])
y_test = test_df['label']
#Identify columns
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
#Apply SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_processed, y_train
)
#Feature Selection
selector = SelectKBest(score_func=mutual_info_classif, k=50)
X_train_selected = selector.fit_transform(X_train_balanced, y_train_balanced)
X_test_selected = selector.transform(X_test_processed)
#Train proposed Random Forest model
rf_proposed = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_proposed.fit(X_train_selected, y_train_balanced)
#Predictions
y_pred = rf_proposed.predict(X_test_selected)
#Evaluation
print("Proposed IML-NID Model Results\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))