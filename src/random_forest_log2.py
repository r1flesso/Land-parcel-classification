from train_val_test_split import X_train, X_val, X_test, y_train, y_val, y_test
from RFC_with_sqrt import RandomForestClassifier_Model
import itertools


# Прописываем списки bootstrap_list и metrics, по которым будем бежать циклом
bootstrap_list = [True, False]
metrics = ['balanced_accuracy', 'f1', 'recall', 'precision', 'cohen_kappa', 'matthews_corrcoef']

if __name__ == '__main__':
    for bootstrap, metric in itertools.product(bootstrap_list, metrics):
        # Создаем экземпляр класса с max_features=None
        Obj = RandomForestClassifier_Model(
            X_train, X_val, X_test, y_train, y_val, y_test, None, bootstrap, metric
        )
        # Вызываем метод для обучения
        Obj.objective_optuna()
