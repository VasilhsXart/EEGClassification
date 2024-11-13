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
import pydot
import graphviz

from collections import Counter
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras import regularizers
from keras.layers import LSTM, Dense, Conv2D, Flatten, Masking, SimpleRNN, Conv1D,MaxPooling1D,MaxPooling2D,BatchNormalization,Input,Reshape,Dropout,SpatialDropout1D,Activation,GlobalAveragePooling1D,SpatialDropout2D
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model

from scipy import stats

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

def data_generator_evaluate(Chunks,Labels, batch_size=16):
    num_chunks = len(Chunks)
    while True:
        indices = np.arange(num_chunks)
        for start in range(0, num_chunks, batch_size):
            end = min(start + batch_size, num_chunks)
            batch_indices = indices[start:end]
            batch_Chunks = Chunks[batch_indices]
            batch_Labels = Labels[batch_indices]
            batch_Chunks = np.transpose(batch_Chunks, (0, 2, 1))
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

# Label_Dictionary ={
#     0: "RestEyesOpen/",
#     1: "RestEyesClosed/",
#     2: "RestAction/",
#     3: "RestImagination/",
#     4: "LeftHandAction/",
#     5: "LeftHandImagination/",
#     6: "BothHandsAction/",
#     7: "BothHandsImagination/",
#     8: "RightHandAction/",
#     9: "RightHandImagination/",
#     10: "BothFeetAction/",
#     11: "BothFeetImagination/"
# }

Label_Dictionary ={
    0: "LeftHandAction",
    1: "RightHandAction/",
    # 2: "BothFeetAction/",
    # 3: "BothHandsAction/"
}

#Root path to the data
root = "ChunksChannelsICA/"
channels = 17

#The labels we choose to work with (4 == Ignore the rest)
start_index = 0
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
            #else:
                #print ("Problematic array")

#Clear some space
del data
del file

sizes = []
counter = 0
#Pad the Chunks to the maximum sample --> 64 x 420 to 64 x max_samples
for i in range(len(Chunks)):
    if(len(Chunks[i][1])<max_samples):
        Chunks[i] = np.pad(Chunks[i], ((0, 0), (0, max_samples - Chunks[i].shape[1])), mode = 'constant')

    #Normalize each channel within a Chunk using Z-score normalization (Standardize)
    for channel in range(Chunks[i].shape[0]):
        mean = np.mean(Chunks[i][channel])
        std = np.std(Chunks[i][channel])

        if std != 0:  #Avoid division by zero
            Chunks[i][channel] = (Chunks[i][channel] - mean) / std
        else:
            #In case std is zero, just subtract the mean
            Chunks[i][channel] = Chunks[i][channel] - mean

#Convert to numpy array
Chunks = np.array(Chunks)
Labels = np.array(Labels)

#Split data set to Train, Test and Validate
X_train, X_test, Y_train, Y_test = train_test_split(Chunks,Labels, test_size=0.3, random_state=42, stratify=Labels)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.5, random_state=42, stratify=Y_train)

#Clear some space
del Labels
del Chunks

feedforward_model = Sequential([
    Reshape((max_samples, channels, 1),input_shape=(max_samples,channels)),

    Conv2D(filters=32, kernel_size=(10,channels), padding='valid'),
    Activation('elu'),
    MaxPooling2D(pool_size=(4,1)),
    BatchNormalization(),

    SpatialDropout2D(0.5),

    Conv2D(filters=64, kernel_size=(5,1), padding='valid'),
    Activation('elu'),
    MaxPooling2D(pool_size=(4,1)),
    BatchNormalization(),

    Flatten(),

    Dense(units=128, activation='elu'),

    Dropout(0.5),

    Dense(units=labels_amount, activation='softmax')

])

#Batch size to feed the neural network
batch_size = 256

steps_per_epoch = math.ceil(len(X_train)/batch_size)

feedforward_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

feedforward_model.summary()
plot_model(feedforward_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
img = plt.imread('model_plot.png')
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')
plt.show()


validation_steps = math.ceil(len(X_val) / batch_size)

#Train the model using the data generator
history = feedforward_model.fit(data_generator(X_train, Y_train, batch_size=batch_size),steps_per_epoch=steps_per_epoch,epochs=10,verbose=1,validation_data=data_generator_evaluate(X_val, Y_val, batch_size=batch_size),validation_steps=validation_steps)

#Extract accuracy data from the history object
epochs = range(1, len(history.history['accuracy']) + 1)
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

#Plot the training and validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_accuracy, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'ro-', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

#Count the steps needed
steps = math.ceil(len(X_test)/batch_size)

#Print
print("------Evaluation------")
#Evaluate the model
loss, accuracy = feedforward_model.evaluate(data_generator_evaluate(X_test,Y_test,batch_size=batch_size),verbose=1,steps=steps)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

#Predict the outcomes of the model
Y_pred = feedforward_model.predict(data_generator_evaluate(X_test,Y_test, batch_size=batch_size), steps=steps)
Y_pred_labels = np.argmax(Y_pred, axis=1)
Y_true_labels = np.argmax(Y_test, axis=1)

#Plot confusion matrix
cm = confusion_matrix(Y_true_labels, Y_pred_labels)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt='.2f',
            xticklabels=np.arange(labels_amount), yticklabels=np.arange(labels_amount))
plt.xlabel('Predicted labels')
plt.ylabel('Actual labels')
plt.title('Confusion Matrix')
plt.show()
