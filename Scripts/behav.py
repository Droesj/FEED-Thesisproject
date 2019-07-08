# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 2019
@author: Sjoerd Evelo

script to analyze FEED behavioral data

functions
---------
decoding_data_prep:
    load data from specified subjects and sessions
    returns Action Unit and emotional rating Dataframes

AU_decoding:
    decoder to predict emotional categories from AUs
    returns acc/AUC scores and long format DF with AUC scores
    NOTE: uses a manual (slow) OHE method --> maybe delete when AU_decoding_OHE works satisfactory

AU_decoding_OHE:
    decoder to predict emotional categories from AUs
    returns acc/AUC scores and long format DF with AUC scores
    NOTE: new version with sklearn.OneHotEncoder

plot_auc_score:
    plot AUC scores per category using a list of long format DataFrames

"""
from glob import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt


def data_prep(sub_labels, n_sessions=4, task='expressive', redo = True):
    '''
    loads and selects behavioral data from .tsv files
    note: now creates a single Dataframe for each subject (concatenates over sessions)
    
    parameters
    ----------
    sub_lables : list
        list of the labels used in the sub-.. folders
        for instance ['01','02','04']
    n_sessions : int, default = 4
        number of sessions to analyze
    task : str, default = 'expressive'
        name of the task
    redo : Boolean, default = True
        filter for 'redo' in filename for session 2 and 3.
        
    Function returns
    ----------------
    AU_dfs : list
        list of dataframes per subject that has only the AU data
    emo_dfs : list
        list of DataFrames per subject that only includes emotional rating data
    '''
    
    AU_dfs = []
    emo_dfs = []
    sessions = '[1-{}]'.format(n_sessions)
    for sub in sub_labels: #now creates df per sub (maybe also per session?)
        files = (glob('/srv/FEED/behav_data/sub-{}/*ses-{}*{}*run*.tsv'.format(sub,sessions,task)))
        #files = (glob('C:\\Users\\droes\\Documents\\GitHub\\data\\sub-{}\\*ses-{}*{}*run*.tsv'.format(sub,sessions,task)))
        files.sort()
        pprint('found the following files for sub-%s: %s' %(sub,[f for f in files]))
        
        dfs = []
        for i, file in enumerate(files):
            if redo:
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
        #delete NaN, circumplex, 'none' and 'test' trials
        df = df.loc[df.rating_category.notna(),:]
        df = df.loc[df.data_split == 'train']
        df = df.loc[df.rating_category != 'Geen van allen']
        #create seperate emo and AU dfs
        emo_cols = ['rating_category', 'rating_intensity_norm']
        emo_df = df[emo_cols]   
        AU_cols = [col for col in df.columns if 'AU' in col and col != 'N_AUs']
        AU_df = df[AU_cols]
        
        AU_dfs.append(AU_df)
        emo_dfs.append(emo_df)
        print('created DataFrames for sub-%s with %i datapoints' %(sub, AU_df.shape[0]))
    return(AU_dfs, emo_dfs)


def auc_validation_calc(X, y, sub, clf, n_splits=10, n_repeats=10):
    '''
    Uses repeated kfold to establish model performance of AU decoding task expressed in AUROC scores
    OneHotEncoding is used to transform categorical variables to binairy vectors
    
    parameters
    ----------
    X : array like, shape = [n_samples, n_features]
        training vector
    y : array like, shape = [n_samples]
        target vector relative to X. 
        Should be a categorical variable
    sub : str
        subject label
    clf : decoding model
        preferably sklearn model
        should contain the methods .fit &.predict
    n_splits : integer
    
    returns
    -------
    auc_df : Pandas DataFrame
        Dataframe with AUROC scores per category and a column with overal score.
    '''
    
    #init
    ohe = OneHotEncoder(sparse=False)
    rskf = RepeatedStratifiedKFold(n_splits, n_repeats)
    auc_score = np.zeros(n_splits*n_repeats)
    names = sorted(y.unique())
    y = y.values[:,np.newaxis]
    y_star = ohe.fit_transform(y)
    auc_dict = {key:[] for key in names}
    
    #split train set again --> stratified k-fold to optimize trials
    for i, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
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


def create_long_df(auc_df):
    '''
    transform dataframe to long format for plotting purposes
    
    parameters
    ----------
    auc_df : Pandas DataFrame
        dataframe to be transformed
    
    returns
    -------
    auc_long: Pandas DataFrame
        long format dataframe
    '''
    auc_df['id'] = auc_df.index
    auc_long = pd.melt(auc_df,
                       id_vars = 'id',
                       var_name = 'category',
                       value_name = 'auc_score').drop('id', axis = 1)

    return auc_long
    

def plot_auc_score(auc_dfs, subs):
    '''
    plot classification scores from long format dataframes with columns 'auc_score' and 'category'
    
    parameters
    ----------
    auc_dfs : list
        contains long format DataFrames per participant
    subs: list
        contains subject labels
    '''
    #TODO: add titles?
    #fix to work for only one participant
    n = len(auc_dfs)
    data = zip(auc_dfs, subs)
    fig, ax = plt.subplots(ncols=n, sharey=True, sharex=False, figsize = (15, 15))
    sns.set(palette='husl')
    for i, (df,sub) in enumerate(data):
        if n ==1:
            g = sns.boxenplot(x = 'category',y = 'auc_score',data = df, ax=ax)
            for item in g.get_xticklabels():
                item.set_rotation(60)
            ax.set_title('sub-%s:'%(sub))
            ax.plot([-1,7],[0.5,0.5],'k',linewidth = 3.5)
        else:
            g = sns.boxenplot(x = 'category',y = 'auc_score',data = df, ax=ax[i])
            for item in g.get_xticklabels():
                item.set_rotation(60)
            ax[i].set_title('sub-%s:'%(sub))
            ax[i].plot([-1,7],[0.5,0.5],'k',linewidth = 3.5)

    plt.show()
    
def AU_decoding_old(AU_dfs, emo_dfs, n_splits=10,n_repeats=10, **kwargs):
    '''
    Calculates decoding scores per emotional category
    
    Parameters
    ----------
    AU_dfs : list 
        list of DataFrames containing AU values per subject
    emo_dfs : list
        List of Dataframes containing emotional rating scores per subject
        should match in lenght and shapes (of DFs) with AU_dfs
    n_splits : int
        number of splits to be used in stratified K-fold
    n_repeats : int
        Number of repeats in stratified K-fold
    **kwargs : dict
        optional arguments for classifier (LinearSVC)
        
    Return objects
    --------------
    score_list : list 
        contains dictionairies of accuracy scores per participant
    auc_score_list : list 
        contains dictionairies of AUC scores per participant    
    auc_long_list : list
        contains long format DataFrames with AUC scores per category per participant
    
    '''
    # TODO: 
    #- figure out best objects to return
    #- clean up code for unused df_list and order initialization
    
    data = zip(AU_dfs,emo_dfs)
    score_list = []
    auc_score_list = []
    auc_df_list = []
    auc_long_list = []
    
    #loop over subjects
    for AU_df, emo_df in data:
        #initialize objects and classifier
        emotions = ['Boos', 'Bang', 'Blij', 'Verrassing', 'Walging', 'Verdrietig']
        correct_dict = {key:[] for key in emotions}
        auc_dict = {key:[] for key in emotions}
        score = np.zeros(n_splits*n_repeats)
        auc_score = np.zeros(n_splits*n_repeats)
        
        X = AU_df.values
        y = emo_df.rating_category.values
        y_star = pd.get_dummies(y)
        
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
        clf = LinearSVC(**kwargs)
        
        #loop over cross-validations
        for i, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
            clf.fit(X[train_idx], y[train_idx])
            score[i] = (clf.score(X[test_idx], y[test_idx]))
            y_hat = clf.predict(X[test_idx])
            correct = y_hat[y_hat == y[test_idx]]
            
            #extract auc_score
            y_hat_star = pd.DataFrame(np.zeros((len(y_hat),len(y_star.columns))),columns = y_star.columns)
            for j, trial in enumerate(y_hat):
                y_hat_star.loc[j,:] = (trial == y_star.columns).astype(int)
            auc_score[i] = roc_auc_score(y_star.loc[test_idx,:],y_hat_star)
            
            #calculate (auc)scores per category
            labels_y_hat, counts_y_hat = np.unique(correct, return_counts=True)
            labels_y, counts_y = np.unique(y[test_idx], return_counts = True)
            for key in correct_dict:
                auc_dict[key].append(roc_auc_score(y_star.loc[test_idx,key],y_hat_star[key]))
                if key in labels_y_hat:
                    n_stim = counts_y[labels_y==key][0]
                    correct_dict[key].append(counts_y_hat[labels_y_hat == key][0]/n_stim)
                else:
                    correct_dict[key].append(0)
                
        auc_score_dict = {'overall auc score' : np.mean(auc_score)}
        score_dict = {'overall score' : np.mean(score)}
        auc_df = pd.DataFrame(data=auc_dict)
        auc_df['overall AUC score'] = auc_score
        auc_df_list.append(auc_df)
        for k,v in auc_dict.items():
            auc_score_dict[k] = np.mean(v) 
        auc_score_list.append(auc_score_dict)    
        
        for k,v in correct_dict.items():
            score_dict[k] = np.mean(v)
        score_list.append(score_dict)

        auc_df['id'] = auc_df.index
        auc_long = pd.melt(auc_df,
                           id_vars = 'id',
                           var_name = 'category',
                           value_name = 'auc_score').drop('id', axis = 1)
        auc_long_list.append(auc_long)
        

    return score_list, auc_score_list, auc_long_list


def AU_decoding_OHE_old(AU_dfs, emo_dfs, n_splits=10,n_repeats=10, **kwargs):
    '''
    Calculates decoding scores per emotional category (using sklearn OHE)
    
    Parameters
    ----------
    AU_dfs : list 
        list of DataFrames containing AU values per subject
    emo_dfs : list
        List of Dataframes containing emotional rating scores per subject
        should match in lenght and shapes (of DFs) with AU_dfs
    n_splits : int
        number of splits to be used in stratified K-fold
    n_repeats : int
        Number of repeats in stratified K-fold
    **kwargs : dict
        optional arguments for classifier (LinearSVC)
        
    Return objects
    --------------
    score_list : list 
        contains dictionairies of accuracy scores per participant
    auc_score_list : list 
        contains dictionairies of AUC scores per participant    
    auc_long_list : list
        contains long format DataFrames with AUC scores per category per participant
    '''
    #TODO: clean up code for unused df_list
    
    data = zip(AU_dfs,emo_dfs)
    score_list = []
    auc_score_list = []
    auc_df_list = []
    auc_long_list= []

    ohe = OneHotEncoder(sparse=False)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    clf = LinearSVC(**kwargs)

    #loop over subjects
    for AU_df, emo_df in data:
        
        #initialize objects
        score = np.zeros(n_splits*n_repeats)
        auc_score = np.zeros(n_splits*n_repeats)
        X = AU_df.values
        y = emo_df.rating_category
        names = y.unique()
        names.sort()
        y = y.values[:,np.newaxis]
        ohe.fit(y)
        y_star = ohe.transform(y)
        correct_dict = {key:[] for key in names}
        auc_dict = {key:[] for key in names}
        
        #loop over cross-validations
        for i, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
            clf.fit(X[train_idx], y[train_idx].ravel())
            
            score[i] = (clf.score(X[test_idx], y[test_idx]))
            y_hat = clf.predict(X[test_idx])[:,np.newaxis]
            y_hat_star = ohe.transform(y_hat)
            auc_score[i] = roc_auc_score(y_star[test_idx,:],y_hat_star)
            correct = y_hat[y_hat == y[test_idx]]
            
            #calculate (auc)scores per category
            labels_y_hat, counts_y_hat = np.unique(correct, return_counts=True)
            labels_y, counts_y = np.unique(y[test_idx], return_counts = True)
            for i, key in enumerate(names): 
                auc_dict[key].append(roc_auc_score(y_star[test_idx][:,i],y_hat_star[:,i]))
                if key in labels_y_hat:
                    n_stim = counts_y[labels_y==key][0]
                    correct_dict[key].append(counts_y_hat[labels_y_hat == key][0]/n_stim)
                else:
                    correct_dict[key].append(0)

        #store scores in DF and dicts    
        auc_df = pd.DataFrame(data=auc_dict)
        auc_df['overall AUC score'] = auc_score
        auc_df['id'] = auc_df.index
        auc_long = pd.melt(auc_df,
                           id_vars = 'id',
                           var_name = 'category',
                           value_name = 'auc_score').drop('id', axis = 1)
        auc_long_list.append(auc_long)
        
        auc_score_dict = {'overall auc score' : np.mean(auc_score)}
        score_dict = {'overall score' : np.mean(score)}
        for k,v in auc_dict.items():
            auc_score_dict[k] = np.mean(v) 
        auc_score_list.append(auc_score_dict)    
        for k,v in correct_dict.items():
            score_dict[k] = np.mean(v)
        score_list.append(score_dict)

    return score_list, auc_score_list, auc_long_list

