import pandas as pd

# Load the dataset
df = pd.read_parquet("data/train.parquet")

print("\nDataset head:")
print(df.head())
print("\nClass balance:")
print(df['target_move'].value_counts(normalize=True))
print("\nFeature info:")
print(df.info()) 