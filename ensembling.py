import pandas as pd
import numpy as np
import os 
import re


submissions_to_ensemble = './submissions/'

all_files = os.listdir(submissions_to_ensemble)    
csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))

df_ = []

submission_scores_ = np.zeros(len(csv_files))



for i, csv_file in enumerate(csv_files):
    df_ += [pd.read_csv(submissions_to_ensemble + csv_file)]
    submission_score = re.search('submission(.*).csv', csv_file).group(1)
    submission_scores_[i] = float(submission_score) / (10**(len(submission_score)-1))


def weight(score):
    return 1 / (1-score)


weights_ = weight(submission_scores_)

print('Submissions score' , submission_scores_)
print('Weights normalized', weights_ / np.sum(weights_))
print('Min expected score', 1 - 1/(np.sum((weights_)** (len(weights_) +1)) ** (1/ (len(weights_) +1 ))))
print('Max expected score', 1 - 1/np.sum(weights_))



solution = df_[0]

print(solution.describe())

for col in solution.columns:
    if col != 'ID':
        solution[col] = solution[col] * weights_[0]


for i,df in enumerate(df_[1:]):
    for col in solution.columns:
        if col!='ID':
            solution[col] = solution[col] + weights_[i+1]*df[col]

for col in solution.columns:
    if col != 'ID':
        solution[col] = solution[col] / np.sum(weights_)

print(solution.describe())

print('Saving results...')
solution.to_csv('submission.csv', index=False)
print('Done!')
