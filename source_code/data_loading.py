import pandas as pd
#Load training and testing datasets
train_path = "Dataset/UNSW_NB15_training-set.csv"
test_path = "Dataset/UNSW_NB15_testing-set.csv"
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
# Display basic information
print("Training Dataset Shape:", train_df.shape)
print("Testing Dataset Shape:", test_df.shape)
print("\nTraining Dataset Columns:")
print(train_df.columns)
print("\nFirst 5 rows of training data:")
print(train_df.head())