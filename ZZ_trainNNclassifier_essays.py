# -*- coding: utf-8 -*-
"""
Created on Mon May 29 09:54:11 2017

@author: ircing, zajic
"""

from __future__ import print_function
import keras
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import matplotlib.pyplot as plt
import numpy as np

import pickle
from time import strftime
from os import path
from scipy.sparse import csr_matrix
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import (CountVectorizer,TfidfVectorizer)
from sklearn.svm import (LinearSVC, NuSVC, SVC)
from sklearn.linear_model import SGDClassifier
from baseline_util import (get_features_and_labels,
                           write_predictions_file,
                           display_classification_results,
                           write_feature_files,
                           ivectors_dict_to_features,
                           combine_feature_matrices,
                           get_labels,
                           get_features_from_text,
                           write_predictions_file_with_probs,
                           generate_submission_csv)

np.random.seed(1234)
SCRIPT_DIR = path.dirname(path.realpath(__file__))

train_partition_name = "all" #"train"
test_partition_name = "dev" #"dev"

trainNN = [1, 0, 0] #= nn,sgd,nusvc
testNN = [1, 0, 0]  # = nn,sgd,nusvc
BASELINE = 'essays'



labels = get_labels(train_partition_name, test_partition_name, BASELINE)
encoded_train_labels, original_training_labels = labels[0]
encoded_test_labels, original_test_labels = labels[1]


#
# Load essays
#

vectorizer = TfidfVectorizer(input="filename",ngram_range=(1, 2),sublinear_tf=True,max_df=0.5,max_features=30000)
transformer = Normalizer()  # Normalize frequencies to unit length

preprocessor = 'tokenized'
training_and_test_data_essays = get_features_from_text(train_partition_name,
                                                 test_partition_name,
                                                 baseline=BASELINE,
                                                 preprocessor=preprocessor,
                                                 vectorizer=vectorizer,                                                 transformer=transformer)

train_matrix_essays = training_and_test_data_essays[0]
test_matrix_essays = training_and_test_data_essays[1]

#-------------------------------------------------------------------------------------
# -----------------NN classifier... on essays--------------------------------------------
#-------------------------------------------------------------------------------------



num_classes = max(encoded_train_labels) + 1
epochs = 6
batch_size = 32
dropout = 0.75

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(encoded_train_labels, num_classes)
y_test = keras.utils.to_categorical(encoded_test_labels, num_classes)

print("Train KERAS NN classifier... on PI_features essays")

model_path = ('PomocneSoubNeu/NNessey_model1_trigram50tis_{ep}epochs{ba}batch.h5')
train_model_path = model_path.format(partition=train_partition_name, ep = epochs, ba = batch_size)

x_train1 = train_matrix_essays.toarray()
x_test1 = test_matrix_essays.toarray()

x_train1 = x_train1.astype('float32')
x_test1 = x_test1.astype('float32')
print('x_train1 shape:', x_train1.shape)
print(x_train1.shape[0], 'train samples')
print(x_test1.shape[0], 'test samples')

if trainNN[0] == 1:
    model1 = Sequential()  # uz neni jina moznost modelu
    model1.add(Dense(units=300, activation='tanh', input_dim=x_train1.shape[1], name='dense_1'))
    model1.add(Dropout(dropout, name='dropout_1'))
    model1.add(Dense(num_classes, activation='softmax', name='dense_2'))

    #sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model1.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])  # nstaveni ucici algoritmus

    #reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100, min_lr=0.000001)

    history1 = model1.fit(x_train1, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test1, y_test))  # natrenuj  .. v priade nevejde do mpameti ...  misto fit train_on_batch (nutne zabespecit nastaveni trenovani)


    score1 = model1.evaluate(x_test1, y_test, verbose=0)  # vypocitej    print('Test loss:', score1[0])
    print('Test accuracy:', score1[1])

    model1.save(train_model_path)  # creates a HDF5 file 'my_model.h5'

    # # list all data in history
    # print(history1.history.keys())
    # # summarize history for accuracy
    # plt.plot(history1.history['acc'])
    # plt.plot(history1.history['val_acc'])
    # plt.title('model3 accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # # summarize history for loss
    # plt.figure()
    # plt.plot(history1.history['loss'])
    # plt.plot(history1.history['val_loss'])
    # plt.title('model1 loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

else:
    model1 = keras.models.load_model(train_model_path)
    model1.summary()

#------------------------------------------------------------------------------------------
# test  ----------------------------------------------------------------#------------------------------------------------------------------------------------------

if testNN[0] == 1:
    predicted_probs = model1.predict(x_test1)
    predicted = np.argmax(predicted_probs, axis=1)

    display_classification_results(encoded_test_labels, predicted)

    BASELINE = 'essays'
    predictions_file_name = strftime(test_partition_name + "_predictions-ZZ_trainNNclassifier_essays-%Y-%m-%d-%H.%M.%S.csv")
    write_predictions_file_with_probs(predicted,predicted_probs, test_partition_name, predictions_file_name, BASELINE)
    predictions_outfile_name = strftime(test_partition_name + "{test_partition_name}_submission-ZZ_trainNNclassifier_essays-%Y-%m-%d-%H.%M.%S.csv")
    generate_submission_csv(BASELINE, test_partition_name, predicted, predictions_outfile_name)


#-------------------------------------------------------------------------------------
# -----------------NN classifier... on essays--------------------------------------------
#-------------------------------------------------------------------------------------
model_path = ('PomocneSoubNeu/SGDessays.h5')
train_model_path = model_path.format(partition=train_partition_name)
if trainNN[1] == 1:
    print("Training the SGDClassifier...")
    clsf = SGDClassifier(loss='modified_huber', penalty='elasticnet', n_iter=40, n_jobs=5, alpha=1e-4)
    clsf.fit(train_matrix_essays, encoded_train_labels)
    pickle.dump(clsf, open(train_model_path, 'wb'))
else:
    clsf = pickle.load(open(train_model_path, 'rb'))

if testNN[1] == 1:
    predicted = clsf.predict(test_matrix_essays)
    display_classification_results(encoded_test_labels, predicted)

    predictions_file_name = strftime( test_partition_name + "_predictions-ZZ_trainSGDCclassifier_essays-%Y-%m-%d-%H.%M.%S.csv")
    write_predictions_file_with_probs(predicted,predicted_probs, test_partition_name, predictions_file_name, BASELINE)
    predictions_outfile_name = strftime(test_partition_name + "_submission-ZZ_trainSGDCclassifier_essays-%Y-%m-%d-%H.%M.%S.csv")
    generate_submission_csv(BASELINE, test_partition_name, predicted, predictions_outfile_name)

#-------------------------------------------------------------------------------------
# ----------------- NuSVC classifier... on ivec--------------------------------------------
#-------------------------------------------------------------------------------------
# model_path = ('../data/NN/{partition}/NuSVCessays.h5')
# train_model_path = model_path.format(partition=train_partition_name)# if trainNN[2] == 1:
#     print("Training the NuSVClassifier...")
#     clsn = NuSVC(random_state=1234)
#     clsn.fit(train_matrix_essays, encoded_train_labels)
#     pickle.dump(clsn, open(train_model_path, 'wb'))
# else:
#     clsn = pickle.load(open(train_model_path, 'rb'))
#
#if testNN[2] == 1:
    # predicted = clsn.predict(test_matrix_essays)
    # display_classification_results(encoded_test_labels, predicted)
    #
    # predictions_file_name = strftime( test_partition_name + "_predictions-ZZ_trainNuSVCclassifier_essays-%Y-%m-%d-%H.%M.%S.csv")
    # write_predictions_file_with_probs(predicted,predicted_probs, test_partition_name, predictions_file_name, BASELINE)
    # predictions_outfile_name = strftime(test_partition_name + "_submission-ZZ_trainNuSVCclassifier_essays-%Y-%m-%d-%H.%M.%S.csv")
    # generate_submission_csv(BASELINE, test_partition_name, predicted, predictions_outfile_name)