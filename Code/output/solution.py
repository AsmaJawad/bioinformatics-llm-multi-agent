import pandas as pd
import numpy as np

df = pd.read_csv('input/expression_counts.csv', usecols=['gene_id', 'gene_name', 'length', 'control_1', 'control_2', 'control_3', 'treatment_1', 'treatment_2', 'treatment_3'])

df['control_mean'] = df[['control_1','control_2','control_3']].mean(axis=1)
df['treatment_mean'] = df[['treatment_1','treatment_2','treatment_3']].mean(axis=1)

df['log2FC'] = np.log2(df['treatment_mean'] / df['control_mean'])

df = df.sort_values(by='log2FC', ascending=False)

print(df[['gene_name', 'control_mean', 'treatment_mean', 'log2FC']])
