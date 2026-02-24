import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Load dataset
train_df = pd.read_csv("Dataset/UNSW_NB15_training-set.csv")
# Drop unnecessary columns
train_df.drop(columns=['id'], inplace=True)
X = train_df.drop(columns=['label', 'attack_cat'])
y = train_df['label']
# Identify columns
categorical_cols = ['proto', 'service', 'state']
numerical_cols = [col for col in X.columns if col not in categorical_cols]
# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ]
)
X_processed = preprocessor.fit_transform(X)
# Apply SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_processed, y)
# Feature Selection
selector = SelectKBest(score_func=mutual_info_classif, k=50)
X_selected = selector.fit_transform(X_balanced, y_balanced)
print("Feature selection completed!")
print("Original feature count:", X_balanced.shape[1])
print("Selected feature count:", X_selected.shape[1])