import pandas as pd
from sklearn.model_selection import train_test_split
from math import log2


df = pd.read_csv('dataset_C.csv')

X = df.drop(columns=['id_2'])
y = df['id_2']

X_train, X_val_test, y_train, y_val_test = train_test_split(
    X, y, train_size=0.7, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_val_test, y_val_test, test_size=0.66, stratify=y_val_test, random_state=42
)

all_samples_count = len(y)

dict_ = {
    i: log2(1 + all_samples_count / len(df[df['id_2'] == i])) for i in range(3, 6)
}

print(dict_)
