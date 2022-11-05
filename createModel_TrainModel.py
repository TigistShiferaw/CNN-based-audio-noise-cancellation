
import sys
import os
import IPython as IP
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pickle
import helpers
from IPython.display import clear_output, display
import soundfile as sf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from IPython.display import clear_output, display
import math
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime
from keras import backend as keras_backend
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, LeakyReLU, SpatialDropout2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from  tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical,plot_model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint 
from keras.regularizers import l2
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
import keras.backend as K
import tensorflow as tf
from keras.models import load_model

us8k_path = os.path.abspath("C:\\Users\\Tigist\\Downloads\\UrbanSound8K.tar\\UrbanSound8K\\UrbanSound8K")
audio_path = os.path.join(us8k_path, 'audio')
metadata_path = os.path.join(us8k_path, 'metadata/UrbanSound8K.csv')
metadata = pd.read_csv(metadata_path)
features = []
labels = []
frames_max = 0
total_samples = len(metadata)
n_mels=40

for index, row in metadata.iterrows():
    file_path = os.path.join(os.path.abspath(audio_path), 'fold' + str(row["fold"]), str(row["slice_file_name"]))
    class_label = row["class"]
    mels = helpers.get_mel_spectrogram(file_path, 0, n_mels=n_mels)
    num_frames = mels.shape[1]
    features.append(mels)
    labels.append(class_label)
    if (num_frames > frames_max):
        frames_max = num_frames
    
print("Finished: {}/{}".format(index, total_samples))

padded_features = helpers.add_padding(features, frames_max)

X = np.array(padded_features)
y = np.array(labels)
np.save("data/X-mel_spec", X)
np.save("data/y-mel_spec", y)
models_path = os.path.abspath('./models')
data_path = os.path.abspath('./data')

labels = [
        'Air Conditioner',
        'Car Horn',
        'Children Playing',
        'Dog bark',
        'Drilling',
        'Engine Idling',
        'Gun Shot',
        'Jackhammer',
        'Siren',
        'Street Music'
    ]

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session( tf.compat.v1.Session(config=config))
X = np.load("data/X-mel_spec.npy")
y = np.load("data/y-mel_spec.npy")
indexes = []
total = len(metadata)
indexes = list(range(0, total))
random.shuffle(indexes)
test_split_pct = 20
split_offset = math.floor(test_split_pct * total / 100)
test_split_idx = indexes[0:split_offset]
train_split_idx = indexes[split_offset:total]
X_test = np.take(X, test_split_idx, axis=0)
y_test = np.take(y, test_split_idx, axis=0)
X_train = np.take(X, train_split_idx, axis=0)
y_train = np.take(y, train_split_idx, axis=0)
test_meta = metadata.iloc[test_split_idx]
train_meta = metadata.iloc[train_split_idx]

le = LabelEncoder()
y_test_encoded = to_categorical(le.fit_transform(y_test))
y_train_encoded = to_categorical(le.fit_transform(y_train))


num_rows = 40
num_columns = 174 
num_channels = 1

X_train = X_train.reshape(X_train.shape[0], num_rows, num_columns, num_channels)
X_test = X_test.reshape(X_test.shape[0], num_rows, num_columns, num_channels)
num_labels = y_train_encoded.shape[1]

def create_model(spatial_dropout_rate_1=0, spatial_dropout_rate_2=0, l2_rate=0):

    model = Sequential()
    model.add(Conv2D(filters=32, 
                     kernel_size=(3, 3), 
                     kernel_regularizer=l2(l2_rate), 
                     input_shape=(num_rows, num_columns, num_channels)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(SpatialDropout2D(spatial_dropout_rate_1))
    model.add(Conv2D(filters=32, 
                     kernel_size=(3, 3), 
                     kernel_regularizer=l2(l2_rate)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(SpatialDropout2D(spatial_dropout_rate_1))
    model.add(Conv2D(filters=64, 
                     kernel_size=(3, 3), 
                     kernel_regularizer=l2(l2_rate)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(spatial_dropout_rate_2))
    model.add(Conv2D(filters=64, 
                     kernel_size=(3,3), 
                     kernel_regularizer=l2(l2_rate)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    
   
    model.add(GlobalAveragePooling2D())


    model.add(Dense(num_labels, activation='softmax'))
    
    return model

spatial_dropout_rate_1 = 0.07
spatial_dropout_rate_2 = 0.14
l2_rate = 0.0005

model = create_model(spatial_dropout_rate_1, spatial_dropout_rate_2, l2_rate)

adam = Adam(lr=1e-4, beta_1=0.99, beta_2=0.999)
model.compile(
    loss='categorical_crossentropy', 
    metrics=['accuracy'], 
    optimizer=adam)

model.summary()
print(len(X_test),len(y_train_encoded))



#  train

num_epochs = 360
num_batch_size = 128
model_file = 'trained2.hdf5'
model_path = os.path.join(models_path, model_file)


# Save checkpoints
checkpointer = ModelCheckpoint(filepath=model_path, 
                               verbose=1, 
                               save_best_only=True)
start = datetime.now()
history = model.fit(X_train, 
                    y_train_encoded, 
                    batch_size=num_batch_size, 
                    epochs=num_epochs, 
                    validation_split=1/12.,
                    callbacks=[checkpointer], 
                    verbose=1)

duration = datetime.now() - start
model_path = os.path.join(models_path, model_file)
model = load_model(model_path)

helpers.model_evaluation_report(model, X_train, y_train_encoded, X_test, y_test_encoded)

y_probs = model.predict(X_test, verbose=0)
yhat_probs = np.argmax(y_probs, axis=1)
y_trues = np.argmax(y_test_encoded, axis=1)
test_meta['pred'] = yhat_probs

cm = confusion_matrix(y_trues, yhat_probs)
accuracies = helpers.acc_per_class(cm)

pd.DataFrame({
    'CLASS': labels,
    'ACCURACY': accuracies
}).sort_values(by="ACCURACY", ascending=False)

