import pandas as pd
from math import log2


df = pd.read_csv('best_class_metrics.csv')


count_all_samples = df['count'].sum()

log_inverse_frequency_dict = {
    i: log2(1 + count_all_samples / df[df['class'] == i]['count'].iloc[0]) for i in range(1, 7)
}

sqrt_weights = {
    i: 1 / (df[df['class'] == i]['count'].iloc[0] / count_all_samples) ** 0.6 for i in range(1, 7)
}

print(log_inverse_frequency_dict)
print(sqrt_weights)
