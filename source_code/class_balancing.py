import pandas as pd
from imblearn.over_sampling import SMOTE
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
#Identify categorical and numerical columns
categorical_cols = ['proto', 'service', 'state']
numerical_cols = [col for col in X_train.columns if col not in categorical_cols]
#Preprocessing (same as before)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ]
)
X_train_processed = preprocessor.fit_transform(X_train)
#Check class distribution BEFORE SMOTE
print("Class distribution before SMOTE:")
print(y_train.value_counts())
#Apply SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_processed, y_train
)
#Check class distribution AFTER SMOTE
print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_balanced).value_counts())
print("\nClass balancing completed successfully!")
print("Balanced training data shape:", X_train_balanced.shape)