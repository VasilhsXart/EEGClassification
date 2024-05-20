import time
import os
import mne
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from mne.datasets import eegbci
import seaborn as sns
import tensorflow

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv2D, Flatten, Masking, SimpleRNN, Conv1D,MaxPooling1D,MaxPooling2D,BatchNormalization,Input,Reshape,Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

#Generator to feed the data into the model with batches
def data_generator(Chunks, Labels, batch_size=16):
    num_chunks = len(Chunks)
    while True:
        indices = np.arange(num_chunks)
        np.random.shuffle(indices)
        for start in range(0, num_chunks, batch_size):
            end = min(start + batch_size, num_chunks)
            batch_indices = indices[start:end]
            batch_Chunks = Chunks[batch_indices]
            batch_Labels = Labels[batch_indices]
            #Transpose from 64 x max_samples to max_samples x 64
            batch_Chunks = np.transpose(batch_Chunks,(0,2,1))
            yield batch_Chunks, batch_Labels

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

#Root path to the data
root = "Chunks3/"

#The labels we choose to work with (4 == Ignore the rest)
start_index = 4
labels_amount = len(Label_Dictionary) - start_index


Chunks =[]
Labels =[]

#Initialize the labels with their categorical form
Label_classes = np.array([i for i in range(labels_amount)])
Categorical_Labels = to_categorical(Label_classes, num_classes=labels_amount)

max_samples = 0

#Read from every class folder
for i in range(labels_amount):
    folder_path = Label_Dictionary[i+start_index]
    chunk_path = os.path.join(root,folder_path)
    for file in os.listdir(chunk_path):
        file_path = os.path.join(chunk_path,file)
        data = np.load(file_path)
        #For every array in the file
        for key in data:
            array = data[key]
            #Ignore the bad labeled ones
            if array.shape[1] < 1000:
                Chunks.append(array)
                Labels.append(Categorical_Labels[i])
                max_samples = max(max_samples, array.shape[1]) if array.shape[1] > max_samples else max_samples
            else:
                print ("Problematic array")

#Clear some space
del data
del file

#Pad the Chunks to the maximum sample --> 64 x 420 to 64 x max_samples
for i in range(len(Chunks)):
    Chunks[i] = np.pad(Chunks[i], ((0, 0), (0, max_samples - Chunks[i].shape[1])), mode = 'constant')


#Convert to numpy array
Chunks = np.array(Chunks)
Labels = np.array(Labels)

#Standardize
Chunks = (Chunks - np.mean(Chunks, axis=2)[:, :, np.newaxis]) / np.std(Chunks, axis=2)[:, :, np.newaxis]
#Normalize
#Chunks = (Chunks - np.min(Chunks, axis=2)[:, :, np.newaxis]) / (np.max(Chunks, axis=2)[:, :, np.newaxis] - np.min(Chunks, axis=2)[:, :, np.newaxis])

X_train, X_test, Y_train, Y_test = train_test_split(Chunks,Labels, test_size=0.1, random_state=42)

#Clear some space
del Labels
del Chunks

#LSTM model
recurrent_model = Sequential([
    Input(shape=(max_samples, 64)),
    LSTM(units=32, return_sequences=False),
    Flatten(),
])

#Configure the input for the CNN
feedforward_input_shape = recurrent_model.output_shape[1:]

#CNN model
feedforward_model = Sequential([
    Input(shape=feedforward_input_shape),
    Reshape((feedforward_input_shape[0], 1)),
    Conv1D(filters=128, kernel_size=5, activation='elu'),
    MaxPooling1D(pool_size=2),
    #Dropout(0.5),
    Conv1D(filters=64, kernel_size=5, activation='elu'),
    MaxPooling1D(pool_size=2),
    #Dropout(0.5),
    Flatten(),
    Dense(units=64, activation='elu'),
    #Dropout(0.5),
    Dense(units=labels_amount, activation='softmax')
])

#Combine the models
combined_model = Sequential([recurrent_model,feedforward_model])

#Batch size to feed the neural network
batch_size = 1
steps_per_epoch = math.ceil(len(X_train)//batch_size)

combined_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Train the model using the data generator
history = combined_model.fit(data_generator(X_train, Y_train, batch_size=batch_size),steps_per_epoch=steps_per_epoch,epochs=10,verbose=1)

#Count the steps needed
steps = math.ceil(len(X_test)/batch_size)

#Evaluate the model
loss, accuracy = combined_model.evaluate(data_generator(X_test, Y_test, batch_size=batch_size),verbose=1,steps=steps)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

#Predict the outcomes of the model
Y_pred = combined_model.predict(data_generator(X_test, Y_test, batch_size=batch_size), steps=steps)
Y_pred_labels = np.argmax(Y_pred, axis=1)
Y_true_labels = np.argmax(Y_test, axis=1)

#Plot confusion matrix
cm = confusion_matrix(Y_true_labels, Y_pred_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=np.arange(labels_amount), yticklabels=np.arange(labels_amount))
plt.xlabel('Predicted labels')
plt.ylabel('Correct labels')
plt.title('Confusion Matrix')
plt.show()
