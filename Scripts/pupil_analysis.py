# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 15:54:04 2019
@author: Sjoerd Evelo
-- for FEED project!--

(mini)library that utilizes hedfpy and ndconv to preprocess and analyze eyetracker .edf files


functions
---------

pupil_preproc: 
    preprocesses .edf files, outputs .hdf5 and .msg files

pupil_response_mapper:
    combines signal and event data, analyses impulse response to events
    outputs combined dataframe, event dataframe and response function object

"""

import nideconv
import hedfpy

from re import split
import os
import pandas as pd
import matplotlib.pyplot as plt


def pupil_preproc(edf,output_dir,**params):
    
    '''
    Pupil signal preprocess script
    
    Parameters
    ----------
    edf : str
        Filepath to the to be processed EDF file
    
    output_dir : str
        Folder where processed HDF5 file will be stored. 
        Will be created if it does not exist yet.
        .msg, .gaz, .gazgz files are stored in the .edf folder <-- check if i can change this!
    
    **params : dict 
        can (must?) include the following keys:
        'sample_rate', 'pupil_lp', 'pupil_hp', 
        'normalization', 'regress_blinks', 'regress_sacs', 
        'use_standard_blinksac_kernels'.
    
    
    Function returns two objects
    hdf5_fn: filepath to HDF5 file
    msg_fn: filepath to .msg_file

    '''

    fn = edf.split('/')[-1]
    base_folder = ''.join(split(r'(/)', edf)[:-2])
    hdf5_fn = os.path.join(output_dir,fn.split('edf')[0] + 'hdf5')
    msg_fn = edf.split('edf')[0] + 'msg'
    
    if not os.path.exists(output_dir):
        print('creating Preproc folder')
        os.makedirs(output_dir)
        
    if os.path.isfile(hdf5_fn):
        print('detected an existing HDF5 file for this edf file, deleting..')
        os.remove(hdf5_fn)
        
    
    print('creating HDF5 file in %s' %(output_dir))
    ho = hedfpy.HDFEyeOperator(hdf5_fn)
    ho.add_edf_file(edf)
    ho.edf_message_data_to_hdf()
    ho.edf_gaze_data_to_hdf(**params)
    print('Preprocessing done!')
    return(hdf5_fn, msg_fn)
    

def pupil_response_mapper(hdf5_fn, msg_fn, include_info=False, left_eye=True , 
                          xlim = (10, 100), **kwargs):
    
    '''
    Impulse response mapper

    parameters
    ----------
    hdf5_fn : str
        Filepath to .hdf5 file containing the to preprocessed eyetracker signal

    msg_fn : str
        Filepath to corresponding .msg file

    include_info : Boolean, default = False
        If True, trail parameter info is extracted from the message file
        and included in the df_c  and events dataframes

    left_eye : boolean, default = True
        If True, left eye 'L_pupil_bp_clean_psc'  is selected for analysis,
        otherwise 'R_pupil_bp_clean_psc' is selected

    **kwargs : dict
        can included following arguments: 
        - xlim : tuple, set limits for signal plot in milliseconds:  (default = (10000,100000))
        - basis_set_method : str (default = 'fourier')
        - interval : list, two values that determine the interval over which the events are fitted.

    Function returns three objects:
    df_c: pandas DataFrame, includes all data (singal and events)
    events: pandas Dataframe, only includes event data
    rf: ResponseFunction object

    '''
    
    if left_eye:
        signal = 'L_pupil_bp_clean_psc'
    else:
        signal = 'R_pupil_bp_clean_psc'
    
    basis_set = kwargs.get('basis_set_method', 'fourier')
    
    if 'expressive' in hdf5_fn:
        task_label = 'expressive'
    else:
        task_label = 'neutral'
    sub_label = hdf5_fn.split('/')[-1][:6]
        
    #Load .msg file
    print('Reading .msg file...')
    with open(msg_fn, 'r') as msg:
        lines = msg.readlines()
        for i, line in enumerate(lines):
            if line[:6] == 'EVENTS':
                sample_rate = float(line.split('\t')[4])
                start = i
                break

    #Load HDF5 file, set and convert index to seconds from start of recording.
    block = str(''.join((split(r'(sub)',hdf5_fn)[-2:])).replace('hdf5', 'edf') + '/block_0')
    df = pd.read_hdf(hdf5_fn,block)
    df = df.set_index('time')
    df.index = df.index / sample_rate  # convert to seconds
    start_rec = df.index[0]
    df.index -= start_rec
    print('Succesfully loaded HDF5 file with %i timepoints and %i columns' %(df.shape[0], df.shape[1]))
    
    #Loop over trial info and store in dataframes
    msgs = [l for l in lines[start+1:] if 'MSG' in l]  
    trial_nr = 1
    dfs = []
    while True:
        these_msg = [l for l in msgs if 'trial %i ' % trial_nr in l]
        if not these_msg:
            break
        onset = float(these_msg[0].split('\t')[1]) / sample_rate - start_rec
        if include_info:
            info = {}
            these_msg = [m for m in these_msg if 'parameter' in m]
            for msg in these_msg:
                tmp = msg.split('parameter\t')[1]
                info[tmp.split(' : ')[0]] = tmp.split(' : ')[1].replace('\n', '')
            dfs.append(pd.DataFrame(info, index=[onset]))
        if not include_info:
            trial_type_msg = [m for m in these_msg if 'trial_type' in m]
            trial_type = trial_type_msg[0].split(' : ')[1].replace('\n', '')
            dfs.append(pd.DataFrame({'trial_type':trial_type,
                                    'trial_nr' : trial_nr}, index=[onset]))
        trial_nr += 1
    
    #create event DataFrame
    events = pd.concat(dfs)
    for col in events.columns:
        events[col] = pd.to_numeric(events[col], errors='ignore')
        msgs = [l for l in lines[start+1:] if 'MSG' in l]
    
    #Concatenate with main dataframe
    df_c = pd.concat((df, events), axis=1)
    for col in df_c.columns:
        if pd.api.types.is_numeric_dtype(df_c[col]):
            df_c[col] = df_c[col].fillna(0)
            
    #Plot first 100 seconds of signal and events
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df_c[signal], c='b')
    ax.plot(df_c.index, [10 if pd.notna(t) else 0 for t in df_c.trial_type], c='r', ls='--')
    ax.set_xlim(xlim)
    ax.set_ylim(-25,25)
    plt.show()    
    
    #Fit response function
    print('Fitting response function...')
    
    rf = nideconv.ResponseFitter(input_signal=df_c[signal],
                                 sample_rate=sample_rate,
                                 oversample_design_matrix=1)
    rf.add_event('stim',
                 onset_times=events.index,
                 basis_set=basis_set,
                 interval=kwargs.get('interval',[0.,2.5]),
                 n_regressors= kwargs.get('n_regeressors', 15),
                 )
    rf.regress()
    rf.plot_timecourses()
    plt.suptitle('Linear deconvolution using GLM and %s' %(basis_set))
    plt.title('%s - %s' %(sub_label, task_label))
    plt.legend()
    plt.show()
    
    print('Analysis done for %s task-%s!' %(sub_label, task_label))
    return(df_c, events, rf)