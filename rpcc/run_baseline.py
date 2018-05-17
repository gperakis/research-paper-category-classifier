import pandas as pd
import os
from rpcc import DATA_DIR

'sample_submission_bow'
df1 = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission_text.csv'))
df2 = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission_graph.csv'))
df3 = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission_bow.csv'))

df = pd.DataFrame(df1['Article'])

for col in df1.columns[1:]:
    df[col] = df1[col] + df2[col] + df3[col]

df = df.div(df.sum(axis=1), axis=0)
df['Article'] = df1['Article']

print(df)
df.to_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), encoding='utf-8', index=False)
