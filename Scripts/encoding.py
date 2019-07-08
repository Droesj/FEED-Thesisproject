# -*- coding: utf-8 -*-
"""
Created on Wed May 22 2019
@author: Sjoerd Evelo

script for FEED encoding analyses

functions
---------
load_and_create obj:
    glob files from '/srv/FEED/mri_data/bids/' 
    indicate subjects, task and space labels
    returns img filepaths, event dataframes, confound dataframes, mask filepaths and json file

filter_event_type:
    filter the event dataframes determined by certain features from the event_type column.
    
    indicate if the filter list needs to be kept or deleted
"""
import nibabel as nib
import numpy as np
import pandas as pd
from glob import glob
from pprint import pprint
from nilearn import plotting
def load_and_create_obj(sub,task,space, mask = 'brain',skip_missing = False):
    '''
    '''
    bids_path = '/srv/FEED/mri_data/bids'
    der_path = '/derivatives/fmriprep'
    
    json = glob(bids_path + '/sub-{0}/ses-*/func/sub-{0}_ses-*_task-{1}_run-1_bold.json'.format(sub,task))[0]
    json = pd.read_json(json)
    imgs_f = glob(bids_path + der_path + '/sub-{0}/ses-*/func/sub-{0}_ses-*_task-{1}_run-*_space-{2}_desc-preproc_bold.nii.gz'.format(sub,task,space))
    imgs_f.sort()
    events_f = sorted(glob(bids_path + '/sub-{0}/ses-*/func/sub-{0}_ses-*_task-{1}_run-*_events.tsv'.format(sub,task)))
    conf_f = sorted(glob(bids_path + der_path + '/sub-{0}/ses-*/func/sub-{0}_ses-*_task-{1}_run-*_desc-confounds_regressors.tsv'.format(sub,task)))

    if mask == 'func':
        mask_f = '/home/c10370862/FEED_new/files/masks/sub-{}_sts_func_mask.nii.gz'.format(sub)
    elif mask == 'dseg':
        mask_f = glob(bids_path + der_path + '/sub-{0}/ses-*/func/sub-{0}_ses-*_task-{1}_run-1_space-{2}_desc-aparcaseg_dseg.nii.gz'.format(sub,task,space))[0]
    elif mask == 'brain':
        mask_f = glob(bids_path + der_path + '/sub-{0}/ses-*/func/sub-{0}_ses-*_task-{1}_run-1_space-{2}_desc-brain_mask.nii.gz'.format(sub,task,space))[0]
    elif mask == 'fdr':
        mask_f = '/home/c10370862/FEED_new/files/masks/sub-{}_STG_func_mask_fdr.nii.gz'.format(sub)
        
        
    if skip_missing:
        events_f = [e for e in events_f if 'sub-13_ses-face3_task-expressive_run-6' not in e]
        events_f = [e for e in events_f if 'sub-04_ses-face1_task-expressive_run-1' not in e]
        events_f = [e for e in events_f if 'sub-04_ses-face1_task-expressive_run-2' not in e]

        conf_f = [c for c in conf_f if 'sub-13_ses-face3_task-expressive_run-6' not in c]
        conf_f = [c for c in conf_f if 'sub-04_ses-face1_task-expressive_run-1' not in c]
        conf_f = [c for c in conf_f if 'sub-04_ses-face1_task-expressive_run-2' not in c]
    
    files = zip(imgs_f, events_f, conf_f)

    models_imgs = []
    models_events = []
    models_conf = []
    for img, event, conf in files:
        pprint('loading files %s, %s, %s' %(img, event, conf))
        models_imgs.append(img)
        models_events.append(pd.read_csv(event,sep='\t'))
        models_conf.append(pd.read_csv(conf,sep='\t'))
    return models_imgs, models_events, models_conf, mask_f, json

def filter_event_type(events, filter_list, keep = False):
    '''
    '''
    for ind, e in enumerate(events):
        for f in filter_list:
            if keep == False:
                e = e[e.event_type != f]
            else:
                e = e[e.event_type == f]
        events[ind] = e
            
    return events

def rsq_calc(y, y_hat):

    mpred = y_hat - y_hat.mean(axis=0)
    my = y - y.mean(axis=0)

    rsq_num = np.sum((my - mpred) ** 2, axis = 0)  
    rsq_den = np.sum((my - np.mean(my, axis = 0)) ** 2, axis = 0)
    rsq = 1 - rsq_num / rsq_den
    
    return rsq

def ROI_mask(mask_f, area):
    '''
    '''
    data = nib.load(mask_f)
    mask_left = data.get_data() == area[0]
    mask_right = data.get_data() == area[1]
    mask_STG = np.logical_or(mask_left, mask_right)
    mask = nib.Nifti1Image(mask_STG, data.affine, data.header)
    return mask

def create_func_mask(nimg, threshold = 0):
    '''
    '''
    data = nimg.get_data()
    data[data<threshold] = 0
    mask_array = np.array(data, dtype=bool)
    func_mask = nib.Nifti1Image(mask_array, nimg.affine, nimg.header)
    
    return func_mask
