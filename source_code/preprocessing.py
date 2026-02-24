import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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
# Identify categorical and numerical columns
categorical_cols = ['proto', 'service', 'state']
numerical_cols = [col for col in X_train.columns if col not in categorical_cols]
# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ]
)
# Apply preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
print("Preprocessing completed successfully!")
print("Training data shape after preprocessing:", X_train_processed.shape)
print("Testing data shape after preprocessing:", X_test_processed.shape)