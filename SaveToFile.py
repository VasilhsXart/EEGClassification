import time
import os
import mne
import json
import numpy as np
import random
import math

#Label == 1 / RestEyesOpen
#Label == 2 / RestEyesClosed
#Label == 3 / RestAction
#Label == 4 / RestImagination
#Label == 5 / LeftHandAction
#Label == 6 / LeftHandImagination
#Label == 7 / BothHandsAction
#Label == 8 / BothHandsImagination
#Label == 9 / RightHandAction
#Label == 10 / RightHandImagination
#Label == 11 / BothFeetAction
#Label == 12 / BothFeetImagination

def extract_labels(file_name, annotation):
    subject_id, recording_id = file_name.split("R")
    key = recording_id[:2]
    label = 0
    if(annotation == "T0"):
        if(int(key) == 1):
            label = 1 #RestEyesOpen
        elif(int(key) == 2):
            label = 2 #RestEyesOpen
        elif(int(key) in [3,5,7,9,11,13]):
            label = 3 #RestAction
        else:
            label = 4 #RestImagination
    elif(annotation == "T1"):
        if(int(key) in [3,7,11]):
            label = 5 #LeftHandAction
        elif(int(key) in [4,8,12]):
            label = 6 #LeftHandImagination
        elif(int(key) in [5,9,13]):
            label = 7 #BothFistsAction
        else:
            label = 8 #BothFistsImagination
    else:
        if(int(key) in [3,7,11]):
            label = 9 #RightHandAction
        elif(int(key) in [4,8,12]):
            label = 10 #RightHandImagination
        elif(int(key) in [5,9,13]):
            label = 11 #BothFeetAction
        else:
            label = 12 #BothFeetImagination
    return label

#Ignore reading file messages
mne.set_log_level('WARNING')

#Root Directories
EEG_dir = "EEGFiles/"
Chunks_dir = "ChunkChannels/"
#Labels_dir = "Labels/"

#Chunk Directories
All_Labels_dir = ["RestEyesOpen/","RestEyesClosed/","RestImagination/"
                  ,"RestAction/","LeftHandAction/","LeftHandImagination/"
                  ,"RightHandAction/","RightHandImagination/"
                  ,"BothHandsAction/","BothHandsImagination/"
                  ,"BothFeetAction/","BothFeetImagination/"]

#Labels to choose path to save
Label_Dictionary ={
    0: "RestEyesOpen/",
    1: "RestEyesClosed/",
    2: "RestAction/",
    3: "RestImagination/",
    4: "LeftHandAction/",
    5: "LeftHandImagination/",
    6: "BothHandsAction/",
    7: "BothHandsImagination/",
    8: "RightHandAction/",
    9: "RightHandImagination/",
    10: "BothFeetAction/",
    11: "BothFeetImagination/"
}

#Channels selected for classification
Classification_Channels = ['C3..', 'C4..', 'Cz..', 'Cp3.', 'Cp4.', 'Cpz.', 'Fc3.', 'Fc4.', 'Fcz.',
                      'F3..', 'F4..', 'Fz..', 'P3..', 'P4..', 'Pz..', 'T7..', 'T8..']

#Create all Directories
try:
    os.makedirs(Chunks_dir)
except:
    print("The root directories for the data already exist")

try:
    for directory in All_Labels_dir:
        os.makedirs(os.path.join(Chunks_dir,directory))
except:
    print("The label and chunk directories already exist")

#Read every edf file for every subject
for folder in os.listdir(EEG_dir):
    folder_path = os.path.join(EEG_dir, folder)
    if os.path.isdir(folder_path):
            print("Working on : ", folder_path)
            Labels_Used = []
            Chunks = [[] for _ in range(12)]
            for file in os.listdir(folder_path):
                if file.endswith('.edf'):
                    file_path = os.path.join(folder_path, file)
                    rawEDF = mne.io.read_raw_edf(file_path, preload=True)
                    rawEDF.set_eeg_reference()
                    #Select specific channels
                    #rawEDF.pick_channels(Classification_Channels)

                    #Proccessing the file
                    rawEDF.notch_filter(50)
                    rawEDF.filter(l_freq=0.0, h_freq=40.0)
                    rawEDF.resample(160, npad='auto')

                    #No bad files exist
                    #rawEDF.interpolate_bads(reset_bads=True)

                    #Read the annotations of the signal and split based on duration
                    rawEvents = mne.events_from_annotations(rawEDF)
                    durations = rawEDF.annotations.duration
                    annotations = rawEDF.annotations.description
                    tmin = 0
                    for i in range(len(durations)):
                        try:
                            chunk = rawEDF.copy().crop(tmin, tmin+durations[i])
                            tmin += durations[i]
                            chunk_data = chunk.get_data()
                            label = extract_labels(file_path,annotations[i])-1
                            if label not in Labels_Used:
                                Labels_Used.append(label)
                            Chunks[label].append(chunk_data)
                        except:
                            chunk = rawEDF.copy().crop(tmin, tmin+durations[i] - 0.01)
                            chunk_data = chunk.get_data()
                            label = extract_labels(file_path,annotations[i])-1
                            if label not in Labels_Used:
                                Labels_Used.append(label)
                            Chunks[label].append(chunk_data)
                            
            print("Saving to file")

            #Save every file depending on the list position it has (List position = Label)
            for label in Labels_Used:
                label_file = Label_Dictionary[label]
                label_file_path = os.path.join(Chunks_dir,label_file)
                json_file = folder + ".npz"
                chunk_file_path = os.path.join(label_file_path,json_file)
                print(chunk_file_path)
                try:
                    # Try to load the existing .npz file
                    data = np.load(chunk_file_path)
                    existing_data = {key: data[key] for key in data}
                except:
                    # If the file doesn't exist, create a new one
                    chunk = Chunks[label][0]
                    Chunks[label].pop(0)
                    np.savez(chunk_file_path, chunk1=chunk)
                    data = np.load(chunk_file_path)
                    existing_data = {key: data[key] for key in data}
                with data as data:
                    for i, chunk in enumerate(Chunks[label]):
                        key = f"chunk{i+2}"
                        existing_data[key] = chunk
                np.savez(chunk_file_path, **existing_data)
