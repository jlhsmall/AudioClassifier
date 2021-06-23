import warnings
warnings.filterwarnings("ignore")

import re
import os
os.environ['TF_KERAS'] = '1'
import random
from datetime import datetime

import librosa
from scipy.io import wavfile
import numpy as np
import pandas as pd
import sklearn as sk
import torch
from torch.utils import data
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.optimizers import SGD
import IPython.display as ipd
import matplotlib.pyplot as plt
plt.style.use('ggplot')


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

### Loading Test/Training Data ###
load_urbansound_data = True # <-- Note: Urbansound8k has a shortcut for testing/debugging, only loads 1 folder (800 instead of 8000)

trimmed_only = False
# Only use wav files that end with extension '-processed.wav'


### Loading/Saving Model ###
save_numpy = 'urban-sounds' # False = do nothing. String = Saves only features/labels into a .npy file within /data directory
load_numpy =  False# False = load direct from wav files (SLOW). String = load features/labels from a .npy file
# save_numpy = False # all_data_<name.npy> and all_labels_<name.npy>
# load_numpy = "kaggle50.npy" # all_data_<name.npy> and all_labels_<name.npy>

load_model_file = False #"all_data_attempt" # False = create new model, String = load pre-trian model from filename (no filename extension).
save_model_file = False #"all_data_attempt" # False = do nothing, String = save trained model to filename (no filename extension).
# save_model_file = "urbansound_first_try"
# load_model_file = "urbansound_first_try"

test_size = 0.2
epochs    = 60
batch_size= 4
optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

do_model_analysis = False


def extract_features(file_name):
    """
    Extracts 193 chromatographic features from sound file.
    including: MFCC's, Chroma_StFt, Melspectrogram, Spectral Contrast, and Tonnetz
    NOTE: this extraction technique changes the time series nature of the data
    """
    features = []

    audio_data, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(audio_data))

    mfcc = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
    features.extend(mfcc)  # 40 = 40

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    features.extend(chroma)  # 12 = 52

    mel = np.mean(librosa.feature.melspectrogram(audio_data, sr=sample_rate).T, axis=0)
    features.extend(mel)  # 128 = 180

    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    features.extend(contrast)  # 7 = 187

    # More possible features to add
    #     tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X, ), sr=sample_rate).T,axis=0)
    #     spec_cent = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate).T, axis=0)
    #     spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate).T, axis=0)
    #     rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate).T, axis=0)
    #     zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data).T, axis=0)
    #     features.extend(tonnetz) # 6 = 193
    #     features.extend(spec_cent)
    #     features.extend(spec_bw)
    #     features.extend(rolloff)
    #     features.extend(zcr)

    return np.array(features)

def display_wav(file = None):
    # Displays comparison of loading a wav file via librosa vs via scipy
    if not file:
        print("No wav file to display")
        return
    librosa_load, librosa_sampling_rate = librosa.load(file)
    scipy_sampling_rate, scipy_load = wavfile.read(file)
    print('original sample rate:',scipy_sampling_rate)
    print('converted sample rate:',librosa_sampling_rate)
    print('\n')
    print('original wav file min~max range:',np.min(scipy_load),'~',np.max(scipy_load))
    print('converted wav file min~max range:',np.min(librosa_load),'~',np.max(librosa_load))
    plt.figure(figsize=(12, 4))
    plt.plot(scipy_load)
    plt.figure(figsize=(12, 4))
    plt.plot(librosa_load)


urban_sounds_df = None
env_sounds_df = None
cat_dog_df = None
def load_all_wav_files(load_urbansound=False,
                       trimmed_only=False):
    '''
    Returns two numpy array
    The first is a numpy array containing each audio's numerical features - see extract_features()
    The second numpy array is the array *STRING* of the label.
    (The array indexes align up between the two arrays. data[idx] is classified as labels[idx])
    '''
    one_file = None
    #THIS WILL TAKE A WHILE!!!!!
    all_data = []
    all_labels = []
    all_files = []
    #UltraSound8K
    if load_urbansound:
        print("loading Ultrasound8k")
        # Data Source: https://urbansounddataset.weebly.com/urbansound8k.html
        global urban_sounds_df
        urban_sounds_df = pd.read_csv("./data/UrbanSound8K/metadata/UrbanSound8K.csv")
        for root, dirs, files in os.walk("./data/UrbanSound8K"):
            print(root, str(len(dirs)), str(len(files)), len(all_data))
#SHORTCUT
 #This is in here for quick tests - only loads first Ultrasound8k folder (instead of all of them)
            #if len(all_data) > 0:
                 #break
#END SHORTCUT
            for idx, file in enumerate(files):
                if file.endswith('.wav'):
                    fname = os.path.join(root, file)
                    if(len(all_data) % 100 == 0):
                        print(str(len(all_data)))
                    features = extract_features(fname)
                    if trimmed_only:
                        if file.endswith('-processed.wav'):
                            file = file.replace('-processed', '')
                        else:
                            continue
                    label = urban_sounds_df[urban_sounds_df.slice_file_name == file]["class"].tolist()[0]
#                     if(label == "dog_bark"):
                    all_data.append(features)
                    all_labels.append(label)
                    one_file = fname
                    all_files.append(fname)
#                     display_wav(fname)
#                     break

    return np.array(all_data), np.array(all_labels), all_files, one_file



def analyze_features(all_data, all_labels):
    #seeking only the numeric features from the data
    numeric_features = all_data.select_dtypes(include = [np.number])
    print(numeric_features.dtypes)
    corr = numeric_features.corr()
    print(corr)
#     print(corr['SalePrice'].sort_values(ascending = False)[:5], '\n')
#     print(corr['SalePrice'].sort_values(ascending = False)[-5:])

def plot_history(history = None):
    # Plots accuracy & loss versus epochs
    if not history:
        print("No history to plot")
        return
    fig = plt.figure(figsize=(10,8))
#     fig = plt.figure(figsize=(20,16))
    plt.plot(history.history['loss'], label="Loss")
    plt.plot(history.history['accuracy'], label="Accuracy")
    plt.axis([0,90,0,1.1])
    plt.title("Accuracy and Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Percentage")
    plt.legend()
    plt.show()


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes,
           yticklabels=classes,
           title="Confusion Matrix for Model 5",
           ylabel='True label',
           xlabel='Predicted label')
    ax.set_ylim(len(classes)-0.5, -0.5)
    ax.set_aspect('auto')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.grid(None)
    plt.show()
def plot_confusion_matrix2(cm):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes,
           yticklabels=classes,
           title="Confusion Matrix for Model 10",
           ylabel='True label',
           xlabel='Predicted label')
    ax.set_ylim(len(classes)-0.5, -0.5)
    ax.set_aspect('auto')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.grid(None)
    plt.show()

all_data = np.array([])
all_labels = np.array([])
one_file = None
if not load_numpy:
    all_data, all_labels, all_files, one_file = load_all_wav_files(load_urbansound_data,
                                          trimmed_only)
else:
    all_data = np.load("data/all_data_"+load_numpy)
    all_labels = np.load("data/all_labels_"+load_numpy)
    all_files = np.load("data/all_files_"+load_numpy)
    one_file = np.load("data/one_file_"+load_numpy)
if save_numpy:
    np.save("data/all_data_"+save_numpy, all_data)
    np.save("data/all_labels_"+save_numpy, all_labels)
    np.save("data/all_files_"+save_numpy, all_files)
    np.save("data/one_file_"+save_numpy, one_file)
print(all_data.shape)
classes = list(set(all_labels)) # classes = unique list of labels
n_classes = len(classes)
numeric_labels = np.array([classes.index(label) for label in all_labels]) # labels by index
print(classes)
#all_data_copy = all_data.copy()
#all_labels_copy = all_labels.copy()

for entry in classes:
    print(entry)


def get_unique_labels(in_labels):
    temp_df = pd.DataFrame({'labels': all_labels})
    temp_df['labels'] = temp_df['labels'].str.lower()

    temp_df.loc[temp_df['labels'] == 'dog_bark'] = 'dog'
    temp_df.loc[temp_df['labels'] == 'bark'] = 'dog'
    temp_df.loc[temp_df['labels'] == 'meow'] = 'cat'
    temp_df.loc[temp_df['labels'] == 'cough'] = 'coughing'
    temp_df.loc[temp_df['labels'] == 'laughing'] = 'laughter'
    temp_df.loc[temp_df['labels'] == 'gun_shot'] = 'gunshot_or_gunfire'
    # Not sure I should do this one
    # temp_df.loc[temp_df['labels'] == 'engine_idling'] = 'engine'
    # temp_df.loc[temp_df['labels'] == 'jackhammer'] = 'drilling'
    # temp_df.loc[temp_df['labels'] == 'water_drops'] = 'pouring_water'
    return temp_df['labels'].to_numpy()

###### COLT REMOVE THIS ######
# all_labels = get_unique_labels(all_labels)
print(all_data.shape)
classes = list(set(all_labels)) # classes = unique list of labels
n_classes = len(classes)
numeric_labels = np.array([classes.index(label) for label in all_labels]) # labels by index
print(classes)

all_data_df = pd.DataFrame(data=all_data[:,:])
all_data_df.insert(0, "WAV", all_files, True)
all_data_df.insert(0, "label", numeric_labels, True)

#all_data_df.set_index('WAV', inplace=True)
print(all_data_df.shape)
print(all_data_df.head())

#urban_sounds_df.head()

models = []
for i in range(10):
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    for j in range(10):
        temp_df = all_data_df[all_data_df['WAV'].str.contains('fold' + str(j + 1))]
        if i == j:
            test_data = temp_df
        else:
            if train_data.empty:
                train_data = temp_df
            else:
                train_data = train_data.append(temp_df)

    x_train = train_data.iloc[:, 2:].to_numpy()
    x_test = test_data.iloc[:, 2:].to_numpy()

    y_train = train_data.iloc[:, [0]].to_numpy()
    y_test = test_data.iloc[:, [0]].to_numpy()

    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=x_train.shape[1]))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))  # sigmoid
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    save_dir = os.path.join(os.getcwd(), 'model')
    filepath = "model-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath),monitor='val_acc', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_test, y_test),
                        verbose=2,
                        callbacks=callbacks_list, validation_split=0.1)
    models.append(
        {'model': model, 'history': history, 'xtrain': x_train, 'ytrain': y_train, 'xtest': x_test, 'ytest': y_test})

total_accuracy = 0
accuracy_avg = 0

total_loss = 0
loss_avg = 0
total_by_class = []
accuracy_dict_sum = {}
cms = []

for i, keyval in enumerate(models):
    print("Model", i + 1, "metrics:")
    model = keyval['model']
    x_train = keyval['xtrain']
    x_test = keyval['xtest']
    y_train = keyval['ytrain']
    y_test = keyval['ytest']
    score_train = model.evaluate(x_train, y_train, verbose=0)
    # print("\tTraining Accuracy:     ", score_train[1])
    # print("\tTraining Cross Entropy: %.2f" % score_train[0])

    score_test = model.evaluate(x_test, y_test, verbose=0)
    print("\tTesting Accuracy:     ", score_test[1])
    print("\tTesting Cross Entropy: %.2f" % score_test[0])

    total_accuracy += score_test[1]
    total_loss += score_test[0]

    # Create a confusion matrix for each model
    y_pred_percentages = model.predict(x_test)  # predicted percentages
    y_pred = np.argmax(y_pred_percentages, axis=1)  # Most prevalent prediction
    cm = confusion_matrix(y_test, y_pred)
    cms.append(cm)

    # Aggregrate accuracy per label
    for i, r in enumerate(cm):
        if classes[i] in accuracy_dict_sum:
            accuracy_dict_sum[classes[i]] += r[i] / np.sum(r) * 100
        else:
            accuracy_dict_sum[classes[i]] = r[i] / np.sum(r) * 100
accuracy_avg = total_accuracy / len(models)
loss_avg = total_loss / len(models)
print("Average Accuracy:", accuracy_avg)
print("Average Loss:    ", loss_avg)


def f(v):
    v = v / len(models)
    return v
accuracy_dict = {k: f(v) for k, v in accuracy_dict_sum.items()}

plot_confusion_matrix(cms[4])
plot_confusion_matrix2(cms[9])

accuracy_df = pd.DataFrame(list(accuracy_dict.items()), columns=['label', 'accuracy'])
accuracy_df.sort_values(by=['accuracy'], inplace=True, ascending=True)
accuracy_df.reset_index(inplace=True)
accuracy_df.set_index('label', inplace=True)
accuracy_df.head()


ax = accuracy_df['accuracy'].plot(kind="bar", title='Average Accuracy Per Label', figsize=(15,10), rot=90)
ax.set_xlabel("label")
ax.set_ylabel("accuracy")


guessing_accuracy = 1/len(classes)
print("Total labels:",  accuracy_df.shape[0])
print("Labels with accuracy < 1/len(labels):", accuracy_df[accuracy_df['accuracy'] <= guessing_accuracy].shape[0])
