# -*- coding: utf-8 -*-
# coding: utf-8

from predpriprava import *
from kmeansMethods import UdelejKmeans, UdelejKmeansRefinementAlg
import theano
import numpy as np
from evaluation import evaluateResults
import scipy
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Lambda, RepeatVector, ConvLSTM2D, Embedding
from keras.optimizers import SGD
from keras import losses, optimizers
from numpy import exp, array, random, dot
from keras.layers.recurrent import LSTM
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Input
from keras.models import Model
from keras import backend as K
from keras import metrics
from neuKohen import *
from neuGWR import *
from evaluation import *
from sklearn import preprocessing

# The Sigmoid function, which describes an S shaped curve.
# We pass the weighted sum of the inputs through this function to
# normalise them between 0 and 1.
def __sigmoid(x):
    print x
    print 1.0 / (1.0 + exp(-x))
    time.sleep(100)
    return 1.0 / (1.0 + exp(-x))


# The derivative of the Sigmoid function.
# This is the gradient of the Sigmoid curve.
# It indicates how confident we are about the existing weight.
def __sigmoid_derivative(x):
    return x * (1.0 - x)


# We train the neural network through a process of trial and error.
# Adjusting the synaptic weights each time.
def train(training_set_inputs, training_set_outputs, number_of_training_iterations, synaptic_weights):
    for iteration in xrange(number_of_training_iterations):
        # Pass the training set through our neural network (a single neuron).
        output = think(training_set_inputs, synaptic_weights)

        # Calculate the error (The difference between the desired output
        # and the predicted output).
        error = training_set_outputs - output

        # Multiply the error by the input and again by the gradient of the Sigmoid curve.
        # This means less confident weights are adjusted more.
        # This means inputs, which are zero, do not cause changes to the weights.
        adjustment = dot(training_set_inputs.T, dot(error, __sigmoid_derivative(output)))

        # Adjust the weights.
        synaptic_weights += adjustment
    return synaptic_weights


# The neural network thinks.
def think(inputs, synaptic_weights):
    # Pass inputs through our neural network (our single neuron).
    return __sigmoid(dot(inputs, synaptic_weights))






velikostSlovniku = 2000
#jazyk = 'english'
#vstup = 'VstupREUTERS'
#vstup = 'Vstup3raw'
#vstup = 'Vstup3raw10NG'
#vstup = 'Vstup3small10'
jazyk = 'czech'
vstup = 'VstupPrepisyVelke'
# vstup = 'VstupRaw'
soubAtextyRaw, soubAslozky = NacteniRawVstupu(vstup)
vycisteneTexty, lemmaTexty, tagsTexty = UpravAVysictiTexty(soubAtextyRaw, vstup, jazyk)

# tady se nastaví s čím se bude dále pracovat jestli s cistými texty nebo jejich lemmaty nebo tagy

vstupPrac = vstup + 'CistyText'
textyPracovni = vycisteneTexty
lemmaTexty = []
tagsTexty = []

slovnik, slovnikPole = VytvorVocab(vstupPrac, velikostSlovniku, textyPracovni)
tfidfMat, nazvySoub = vytvorTFIDF(vstupPrac, textyPracovni, slovnikPole)

pozadVystup = []
propojeni = {}
cisloToShl = {}
cisloShl = 0
for soub in nazvySoub:
    if not propojeni.has_key(soubAslozky[soub]):
        pozadVystup.append(cisloShl)
        propojeni[soubAslozky[soub]] = cisloShl
        cisloToShl[cisloShl] = soubAslozky[soub]
        cisloShl += 1
    else:
        pozadVystup.append(propojeni[soubAslozky[soub]])

velikost, okno, alphaa, minalphaa, minimalniPocetCetnostiSlov = 2000, 0.01, 0.01, 0.01, 5
prvniItr = 1
maxP, maxAlf, maxPref, maxAlfRef = 0.0, 0.0, 0.0, 0.0

maticeDoc2VecVah = VytvorReprDoc2Vec(vstupPrac, textyPracovni, nazvySoub, velikost, okno, alphaa, minalphaa, minimalniPocetCetnostiSlov)


#na zkoušku nejdříve kmeans použité jako vstup y pro trénování neu

# nastavení parametrů kmeans a provedení
maticePouzVah = tfidfMat
maxIter = 10000
tolerancee = 0.0001
nInit = 100
Ncomponents = int(math.pow(len(textyPracovni), (1.0 / (1 + (np.log(len(textyPracovni)) / 10.0)))))

P, R, Plsa, Rlsa = [], [], [], []
for ii in range(2):
    vysledekClusteringu, vysledekClusteringuLSA, clustering, clusteringLSA = UdelejKmeans(vstupPrac+'TFIDF', soubAslozky, maxIter, tolerancee, nInit, Ncomponents, maticePouzVah, ii)


pocShluku = 0
uzJeSl = {}
for keyy in soubAslozky:
    if not uzJeSl.has_key(soubAslozky[keyy]):
        uzJeSl[soubAslozky[keyy]] = soubAslozky[keyy]
        pocShluku += 1
if pocShluku == 1:
    pocShluku = 31
print 'Počet shluků je nastaven na: ' + str(pocShluku)

maticeDoc2VecVah = np.array(maticeDoc2VecVah)
Y = np.array(pozadVystup)
x_train = maticeDoc2VecVah
y_train = keras.utils.to_categorical(Y, num_classes=pocShluku)
x_test = maticeDoc2VecVah
y_test = keras.utils.to_categorical(Y, num_classes=pocShluku)

# ---------------------------- supervised neu ----------------------------------------------------
model = Sequential()
model.add(Dense(320, activation='relu', input_dim=velikost))
model.add(Dropout(0.5))
model.add(Dense(1028, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(pocShluku, activation='softmax'))


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=100,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
print
print score

vys = model.predict(x_test, batch_size=128)

vys = np.argmax(vys,axis=-1)

Pp, Rr = evaluateResults(soubAslozky, vysledekClusteringu, nazvySoub)
print 'K-means:'
print 'P: ' + str(Pp)
print 'R: ' + str(Rr)
Pp, Rr = evaluateResults(soubAslozky, vys, nazvySoub)
print 'Neu natrénovaná podle vstupu (supervised):'
print 'P: ' + str(Pp)
print 'R: ' + str(Rr)



# ---------------------------- unsupervised neu (autoencoder) ----------------------------------------------------

'''
# this is the size of our encoded representations
encoding_dim = 2000
x_train = np.array(maticeDoc2VecVah)
inp = Input(shape=(2000,))
encoded = Dense(encoding_dim, activation='relu')(inp)
decoded = Dense(2000, activation='relu')(encoded)
decoded2 = Dropout(0.5)(decoded)
decoded3 = Dense(pocShluku, activation='softmax')(decoded2)

# this model maps an input to its reconstruction
autoencoder = Model(inp, decoded)
autoencoder.compile(optimizer='adadelta', loss='mse')

autoencoder.fit(x_train, x_train,
                epochs=1,
                batch_size=128)
vys = autoencoder.predict(x_train)
'''



'''
normMatD2V = []
for vec in maticeDoc2VecVah:
    normalized = ((vec-min(vec))/(max(vec)-min(vec)))
    normMatD2V.append(normalized)
normMatD2V = np.array(normMatD2V)
'''


Y = np.array(pozadVystup)
X_train = maticeDoc2VecVah
y_train = keras.utils.to_categorical(Y, num_classes=pocShluku)
X_test = maticeDoc2VecVah
y_test = keras.utils.to_categorical(Y, num_classes=pocShluku)

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 2000
# truncate and pad input sequences
max_review_length = 2000
print X_train.shape
print X_train
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

print X_train.shape
print X_train
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(320, activation='relu'))
model.add(Dense(pocShluku, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, nb_epoch=10, batch_size=64)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

vys = model.predict(X_test, batch_size=64)

print len(vys)
print len(vys[0])
print vys[0]


vys = np.argmax(vys,axis=-1)

print vys
acc = evaluateResultsAcc(soubAslozky, vys, nazvySoub)
print 'LSTM na doc2vec -- Accuracy: ' + str(acc) #+ '; Precision: ' + str(Pp) + '; Recall: ' + str(Rr)
Pp, Rr = evaluateResults(soubAslozky, vys, nazvySoub)
print 'LSTM neu natrénovaná podle vstupu (supervised):'
print 'P: ' + str(Pp)
print 'R: ' + str(Rr)


'''
Y = np.array(pozadVystup)
x_train = maticeDoc2VecVah
y_train = keras.utils.to_categorical(Y, num_classes=pocShluku)
x_test = maticeDoc2VecVah
y_test = keras.utils.to_categorical(Y, num_classes=pocShluku)

model = Sequential()
model.add(Dense(320, activation='relu', input_dim=velikost))
model.add(Dropout(0.5))
model.add(Embedding(64, 128, input_length=320))
model.add(LSTM(5, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(5))  # return a single vector of dimension 32
model.add(Dense(320, activation='linear'))
#model.add(Dropout(0.5))
model.add(Dense(1028, activation='linear'))
#model.add(Dropout(0.5))
model.add(Dense(pocShluku, activation='linear'))
#sgd = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100)

vys = model.predict(x_test, batch_size=128)

vys = np.argmax(vys,axis=-1)

Pp, Rr = evaluateResults(soubAslozky, vys, nazvySoub)
print 'LSTM neu natrénovaná podle vstupu (supervised):'
print 'P: ' + str(Pp)
print 'R: ' + str(Rr)

'''
