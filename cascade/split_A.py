import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('dataset_A.csv')

X = df.drop(columns=['rarity'])
y = df['rarity']

X_train, X_val_test, y_train, y_val_test = train_test_split(
    X, y, train_size=0.7, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_val_test, y_val_test, test_size=0.66, stratify=y_val_test, random_state=42
)
