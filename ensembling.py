import pandas as pd
import os 




submissions_to_ensemble = './submissions_077/'

all_files = os.listdir(submissions_to_ensemble)    
csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))

df_ = []

for csv_file in csv_files:
    df_ += [pd.read_csv(submissions_to_ensemble + csv_file)]

solution = df_[0]

for df in df_[1:]:
    for col in solution.columns:
        if col!='ID':
            solution[col] = solution[col] + df[col]


for col in solution.columns:
    if col !='ID':
        solution[col] = solution[col] / len(df_)

print('Saving results...')
solution.to_csv('submission.csv', index=False)
print('Done!')
