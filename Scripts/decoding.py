from glob import glob
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

def create_emo_df(sub, task):

    files = (glob('/srv/FEED/behav_data/sub-{}/*ses-*_task-{}*run*.tsv'.format(sub,task)))
    files.sort()
    #pprint('found the following files for sub-%s: %s' %(sub,[f for f in files]))
    dfs = []
    for i, file in enumerate(files):
        if 'ses-1_' not in file:
            if 'redo' not in file:
                print('skipping file %s' %(file))
                continue
        print('loading file %s' %(file))
        tmp_df = pd.read_csv(file, sep = '\t', index_col=0)
        tmp_df['new_index'] = tmp_df.index+len(tmp_df)*(i)
        tmp_df = tmp_df.set_index(tmp_df.new_index.values)
        dfs.append(tmp_df)
    df = pd.concat(dfs, sort = False)

    df = df.loc[df.rating_category.notna(),:]
    #df = df.loc[df.rating_category != 'Geen van allen']
    df = df.loc[df.data_split == 'train']
    emo_cols = ['rating_category', 'rating_intensity_norm', 'trial_type']
    emo_df = df[emo_cols]
    return emo_df



def auc_validation_calc(X, y, sub, clf, n_runs):
    #init
    ohe = OneHotEncoder(sparse=False)
    auc_score = np.zeros(n_splits*n_repeats)
    names = sorted(y.unique())
    y = y.values[:,np.newaxis]
    y_star = ohe.fit_transform(y)
    auc_dict = {key:[] for key in names}
    
    groups = np.repeat(range(1,n_runs+1), 29)
    group_kfold = GroupKFold(n_splits=n_runs)

    for train_idx, test_idx in group_kfold.split(X, y, groups):
    #split train set again --> stratified k-fold to optimize trials
        print('loop: %i' %(i))
        clf.fit(X[train_idx], y[train_idx].ravel())
        y_hat = clf.predict(X[test_idx])[:,np.newaxis]
        y_hat_star = ohe.transform(y_hat)
        auc_score[i] = roc_auc_score(y_star[test_idx,:],y_hat_star)
        for i, key in enumerate(names): 
            auc_dict[key].append(roc_auc_score(y_star[test_idx][:,i],y_hat_star[:,i]))
            
    auc_df = pd.DataFrame(data=auc_dict)
    auc_df['overall AUC score'] = auc_score
    mean_score = np.mean(auc_score)
    print('mean AUC score sub-%s: %f' %(sub, mean_score))
    
    return auc_df