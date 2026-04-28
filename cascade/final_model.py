import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score
from train_val_test_split import X_test, y_test


model_A = mlflow.pyfunc.load_model(r'runs:/3e84d4f937e6423eb6b70903b680e8bf/model')     # Тут нужно самостоятельно вставить путь к модели A
model_B = mlflow.pyfunc.load_model(r'runs:/1438d4c6c155405884bf99755d7dd798/model')     # Тут нужно самостоятельно вставить путь к модели B
model_C = mlflow.pyfunc.load_model(r'runs:/60bcfc75742645e0a1b7ef1fcb30512e/model')     # Тут нужно самостоятельно вставить путь к модели C

pred_A = model_A.predict(X_test)
pred_B = model_B.predict(X_test)
pred_C = model_C.predict(X_test)

y_pred_final = np.empty_like(y_test)

mask_not_rare = pred_A == 0
mask_rare = pred_A == 1

y_pred_final[mask_not_rare] = pred_C[mask_not_rare]
y_pred_final[mask_rare] = pred_B[mask_rare]

classes = [1, 2, 3, 4, 5, 6]
recall_values = []
accuracy_values = []

for cls in classes:
    recall = recall_score(
        y_test,
        y_pred_final,
        labels=[cls],
        average=None,
        zero_division=0
    )[0]
    recall_values.append(recall)

    mask_class = y_test == cls
    accuracy = (y_pred_final[mask_class] == y_test[mask_class]).mean()
    accuracy_values.append(accuracy)

# создаём единый DataFrame
df_metrics = pd.DataFrame({
    'class': classes + ['overall'],
    'recall': recall_values + [None],
    'accuracy': accuracy_values + [accuracy_score(y_test, y_pred_final)]
})

df_metrics.to_csv('final_class_metrics.csv', index=False)
