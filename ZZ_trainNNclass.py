# -*- coding: utf-8 -*-
# coding: utf-8
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
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from evaluation import evaluateResultsAcc2, evaluateResultsAcc
from sklearn import svm
from predpriprava import *
from upravaDat import *

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

trainNN = [1, 1, 0] #= nn,sgd,nusvc
testNN = [1, 1, 0]  # = nn,sgd,nusvc
BASELINE = 'essays'

velikostSlovniku = 5000
jazyk = 'czech'
#vstup = 'VstupPrepisy'
#vstup = 'VstupPrepisyVelke'
#vstup = 'VstupPrepisyStereo'
#vstup = 'VstupPrepisyStereoVelke'
#vstup = 'VstupPrepisyStereoAmono'
vstup = 'VstupPrepisyStereoAmonoVelke'

soubAtextyRaw, soubAslozky = NacteniRawVstupu(vstup)
vycisteneTexty, lemmaTexty, tagsTexty = UpravAVysictiTexty(soubAtextyRaw, vstup, jazyk)
'''
print(' '.join(vycisteneTexty['02463']))
print(vycisteneTexty['02463'])
print(lemmaTexty['02463'])
for docc in vycisteneTexty:
    slova = vycisteneTexty[docc]
    slovaVybrane = []
    for slovo in slova:
        if len(slovo) < 2:
            if
'''
vstupPrac = vstup +'Lemma'
textyPracovni = lemmaTexty
velikost, okno, alphaa, minalphaa, minimalniPocetCetnostiSlov, Ncomponents = 500, 0.01, 0.019, 0.019, 5, 200
slovnik, slovnikPole = VytvorVocab(vstupPrac, velikostSlovniku, textyPracovni)
tfidfMat, nazvySoub = vytvorTFIDF(vstupPrac, textyPracovni, slovnikPole)
tfidfMat = tfidfMat.toarray()
maticeDoc2VecVah = VytvorReprDoc2Vec(vstupPrac, textyPracovni, nazvySoub, velikost, okno, alphaa, minalphaa, minimalniPocetCetnostiSlov)

# část na výpočet výsledků s LSA
svd = TruncatedSVD(algorithm='randomized', n_components=Ncomponents, n_iter=10, random_state=None, tol=0.0)
normalizer = Normalizer(norm='l2', copy=False)
lsa = make_pipeline(svd, normalizer)
tfidfMatLSA = lsa.fit_transform(tfidfMat)

# část na výpočet výsledků s LSA
svd = TruncatedSVD(algorithm='randomized', n_components=Ncomponents, n_iter=10, random_state=None, tol=0.0)
normalizer = Normalizer(norm='l2', copy=False)
lsa = make_pipeline(svd, normalizer)
maticeDoc2VecVahLSA = lsa.fit_transform(maticeDoc2VecVah)

maticeTFIDFD2V = np.append(tfidfMatLSA, maticeDoc2VecVahLSA, axis=1)
maticeCelaTFIDFaD2V = np.append(tfidfMat, maticeDoc2VecVah, axis=1)

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

labels = []
for ss in nazvySoub:
    labels.append(soubAslozky[ss])

pocSoub = len(soubAslozky)
procTrain = 70.0
pocCrossValid = 10
pocTrain = int((pocSoub / 100.0) * procTrain)
pocTest = pocSoub - pocTrain
skokPozZacTrain = int(pocSoub / float(pocCrossValid))

pocVshl = {}
for soubb in soubAslozky:
    if pocVshl.has_key(soubAslozky[soubb]):
        pocVshl[soubAslozky[soubb]] = pocVshl[soubAslozky[soubb]] + 1
    else:
        pocVshl[soubAslozky[soubb]] = 1
pocTrainShl = {}
pocTestShl = {}
skokPozZacTrainShl = {}
for shll in pocVshl:
    pocTrainShl[shll] = int((pocVshl[shll] / 100.0) * procTrain)
    pocTestShl[shll] = int((pocVshl[shll] / 100.0) * (100.0 - procTrain))
    skokPozZacTrainShl[shll] = int(pocVshl[shll] / float(pocCrossValid))

epochs = 200
batch_size = 100
dropout = 0.9
dropout2 = 0.9
un = 500
un2 = 300
un3 = 640
un4 = 320

accKer, accKerLSA, accKerdoc2vec, accKerdoc2vecLSA, accKertfidfdoc2vec, accKerCelTFIDFaD2V = [], [], [], [], [], []
accMoje, accMojeLSA, accMojeoc2vec, accMojedoc2vecLSA, accMojetfidfdoc2vec, accMojeCelTFIDFaD2V = [], [], [], [], [], []
accSVM, accSVMLSA, accSVMoc2vec, accSVMdoc2vecLSA, accSVMtfidfdoc2vec, accSVMCelTFIDFaD2V = [], [], [], [], [], []

for crossValid in range(pocCrossValid):
    print('Probíhá ' + str(crossValid) + ' cyklus cross-validace.')
    '''
    pocShluku = 0
    uzJeSl = {}
    for keyy in soubAslozky:
        if not uzJeSl.has_key(soubAslozky[keyy]):
            uzJeSl[soubAslozky[keyy]] = soubAslozky[keyy]
            pocShluku += 1
    num_classes = pocShluku

    pocPozTr = crossValid * skokPozZacTrain
    pocPozTs = pocPozTr + pocTrain
    print
    'Cyklus: ' + str(crossValid)
    if pocPozTs <= pocSoub:
        soubTrain = nazvySoub[pocPozTr:pocPozTs]
        pozadVystupTrain = pozadVystup[pocPozTr:pocPozTs]
        tfidfTrain = tfidfMat[pocPozTr:pocPozTs]
        tfidfLSATrain = tfidfMatLSA[pocPozTr:pocPozTs]
        doc2vecTrain = maticeDoc2VecVah[pocPozTr:pocPozTs]
        doc2vecLSATrain = maticeDoc2VecVahLSA[pocPozTr:pocPozTs]
        maticeTFIDFD2Vtrain = maticeTFIDFD2V[pocPozTr:pocPozTs]
    else:
        konTr = pocTrain - len(nazvySoub[pocPozTr:pocSoub - 1])
        soubTrain = nazvySoub[pocPozTr:pocSoub - 1] + nazvySoub[0:konTr]
        pozadVystupTrain = pozadVystup[pocPozTr:pocSoub - 1] + pozadVystup[0:konTr]
        tfidfTrain = np.append(tfidfMat[pocPozTr:pocSoub - 1], tfidfMat[0:konTr], axis=0)
        tfidfLSATrain = np.append(tfidfMatLSA[pocPozTr:pocSoub - 1], tfidfMatLSA[0:konTr], axis=0)
        doc2vecTrain = maticeDoc2VecVah[pocPozTr:pocSoub - 1] + maticeDoc2VecVah[0:konTr]
        doc2vecLSATrain = np.append(maticeDoc2VecVahLSA[pocPozTr:pocSoub - 1], maticeDoc2VecVahLSA[0:konTr], axis=0)
        maticeTFIDFD2Vtrain = np.append(maticeTFIDFD2V[pocPozTr:pocSoub - 1], maticeTFIDFD2V[0:konTr], axis=0)

    if pocPozTs + pocTest <= pocSoub:
        soubTest = nazvySoub[pocPozTs:pocPozTs + pocTest]
        tfidfTest = tfidfMat[pocPozTs:pocPozTs + pocTest]
        tfidfLSATest = tfidfMatLSA[pocPozTs:pocPozTs + pocTest]
        doc2vecTest = maticeDoc2VecVah[pocPozTs:pocPozTs + pocTest]
        doc2vecLSATest = maticeDoc2VecVahLSA[pocPozTs:pocPozTs + pocTest]
        maticeTFIDFD2VTest = maticeTFIDFD2V[pocPozTs:pocPozTs + pocTest]
        pozadVystupTest = pozadVystup[pocPozTs:pocPozTs + pocTest]
    else:
        konTs = pocTest - len(nazvySoub[pocPozTs:pocSoub - 1])
        soubTest = nazvySoub[pocPozTs:pocSoub - 1] + nazvySoub[0:konTs]
        tfidfTest = np.append(tfidfMat[pocPozTs:pocSoub - 1], tfidfMat[0:konTs], axis=0)
        tfidfLSATest = np.append(tfidfMatLSA[pocPozTs:pocSoub - 1], tfidfMatLSA[0:konTs], axis=0)
        doc2vecTest = maticeDoc2VecVah[pocPozTs:pocSoub - 1] + maticeDoc2VecVah[0:konTs]
        doc2vecLSATest = np.append(maticeDoc2VecVahLSA[pocPozTs:pocSoub - 1], maticeDoc2VecVahLSA[0:konTs], axis=0)
        maticeTFIDFD2VTest = np.append(maticeTFIDFD2V[pocPozTs:pocSoub - 1], maticeTFIDFD2V[0:konTs], axis=0)
        pozadVystupTest = pozadVystup[pocPozTs:pocSoub - 1] + pozadVystup[0:konTs]

    soubAslozkyTest = {}
    for soub in soubTest:
        soubAslozkyTest[soub] = soubAslozky[soub]
    '''


    # TF-IDF matice svm a neu
    tfidfTrain, tfidfTest, pozadVystupTrain, soubAslozkyTest, pozadVystupTest, pocShluku, soubTest = ExtrakceCastiMatice(
        nazvySoub, soubAslozky, pocTrainShl, tfidfMat, pozadVystup, skokPozZacTrainShl, crossValid, propojeni)
    num_classes = pocShluku
    acc = svmCast(tfidfTrain, pozadVystupTrain, tfidfTest, soubAslozkyTest, soubTest)
    accSVM.append(acc)

    accK, accNeu = neuProved(tfidfTrain, tfidfTest, pozadVystupTrain, pozadVystupTest, num_classes, train_partition_name, epochs, batch_size, soubAslozkyTest, soubTest, dropout, dropout2, un, un2)
    accMoje.append(accNeu)
    accKer.append(accK)

    # TF-IDF matice snížená LSA svm a neu
    tfidfTrainLSA, tfidfTestLSA, pozadVystupTrain, soubAslozkyTest, pozadVystupTest, pocShluku, soubTest = ExtrakceCastiMatice(
        nazvySoub, soubAslozky, pocTrainShl, tfidfMatLSA, pozadVystup, skokPozZacTrainShl, crossValid, propojeni)
    num_classes = pocShluku
    acc = svmCast(tfidfTrainLSA, pozadVystupTrain, tfidfTestLSA, soubAslozkyTest, soubTest)
    accSVMLSA.append(acc)

    accK, accNeu = neuProved(tfidfTrainLSA, tfidfTestLSA, pozadVystupTrain, pozadVystupTest, num_classes,
                             train_partition_name, epochs, batch_size, soubAslozkyTest, soubTest, dropout, dropout2, un,
                             un2)
    accMojeLSA.append(accNeu)
    accKerLSA.append(accK)

    # doc2vec matice svm a neu
    doc2vecTrain, doc2vecTest, pozadVystupTrain, soubAslozkyTest, pozadVystupTest, pocShluku, soubTest = ExtrakceCastiMatice(
        nazvySoub, soubAslozky, pocTrainShl, maticeDoc2VecVah, pozadVystup, skokPozZacTrainShl, crossValid, propojeni)
    num_classes = pocShluku
    acc = svmCast(doc2vecTrain, pozadVystupTrain, doc2vecTest, soubAslozkyTest, soubTest)
    accSVMoc2vec.append(acc)

    accK, accNeu = neuProved(doc2vecTrain, doc2vecTest, pozadVystupTrain, pozadVystupTest, num_classes,
                             train_partition_name, epochs, batch_size, soubAslozkyTest, soubTest, dropout, dropout2, un,
                             un2)
    accMojeoc2vec.append(accNeu)
    accKerdoc2vec.append(accK)

    # doc2vec matice snížená LSA svm a neu
    doc2vecTrainLSA, doc2vecTestLSA, pozadVystupTrain, soubAslozkyTest, pozadVystupTest, pocShluku, soubTest = ExtrakceCastiMatice(
        nazvySoub, soubAslozky, pocTrainShl, maticeDoc2VecVahLSA, pozadVystup, skokPozZacTrainShl, crossValid, propojeni)
    num_classes = pocShluku
    acc = svmCast(doc2vecTrainLSA, pozadVystupTrain, doc2vecTestLSA, soubAslozkyTest, soubTest)
    accSVMdoc2vecLSA.append(acc)

    accK, accNeu = neuProved(doc2vecTrainLSA, doc2vecTestLSA, pozadVystupTrain, pozadVystupTest, num_classes,
                             train_partition_name, epochs, batch_size, soubAslozkyTest, soubTest, dropout, dropout2, un,
                             un2)
    accMojedoc2vecLSA.append(accNeu)
    accKerdoc2vecLSA.append(accK)

    # tfidf a doc2vec matice svm a neu
    tfidfdoc2vecTrainLSA, tfidfdoc2vecTestLSA, pozadVystupTrain, soubAslozkyTest, pozadVystupTest, pocShluku, soubTest = ExtrakceCastiMatice(
        nazvySoub, soubAslozky, pocTrainShl, maticeTFIDFD2V, pozadVystup, skokPozZacTrainShl, crossValid,
        propojeni)
    num_classes = pocShluku
    acc = svmCast(tfidfdoc2vecTrainLSA, pozadVystupTrain, tfidfdoc2vecTestLSA, soubAslozkyTest, soubTest)
    accSVMtfidfdoc2vec.append(acc)

    accK, accNeu = neuProved(tfidfdoc2vecTrainLSA, tfidfdoc2vecTestLSA, pozadVystupTrain, pozadVystupTest, num_classes,
                             train_partition_name, epochs, batch_size, soubAslozkyTest, soubTest, dropout, dropout2, un,
                             un2)
    accMojetfidfdoc2vec.append(accNeu)
    accKertfidfdoc2vec.append(accK)

    # tfidf celá a doc2vec matice svm a neu
    tfidfdoc2vecTrainLSA, tfidfdoc2vecTestLSA, pozadVystupTrain, soubAslozkyTest, pozadVystupTest, pocShluku, soubTest = ExtrakceCastiMatice(
        nazvySoub, soubAslozky, pocTrainShl, maticeCelaTFIDFaD2V, pozadVystup, skokPozZacTrainShl, crossValid,
        propojeni)
    num_classes = pocShluku
    acc = svmCast(tfidfdoc2vecTrainLSA, pozadVystupTrain, tfidfdoc2vecTestLSA, soubAslozkyTest, soubTest)
    accSVMCelTFIDFaD2V.append(acc)

    accK, accNeu = neuProved(tfidfdoc2vecTrainLSA, tfidfdoc2vecTestLSA, pozadVystupTrain, pozadVystupTest, num_classes,
                             train_partition_name, epochs, batch_size, soubAslozkyTest, soubTest, dropout, dropout2, un,
                             un2)
    accMojeCelTFIDFaD2V.append(accNeu)
    accKerCelTFIDFaD2V.append(accK)



    '''
    tfidfTrain, tfidfTest, pozadVystupTrain, soubAslozkyTest, pozadVystupTest, pocShluku, soubTest = ExtrakceCastiMatice(
        nazvySoub, soubAslozky, pocTrainShl, tfidfMat, pozadVystup, skokPozZacTrainShl, crossValid, propojeni)
    num_classes = pocShluku

    train_matrix_essays = tfidfTrain
    test_matrix_essays = tfidfTest
    # -------------------------------------------------------------------------------------
    # -----------------NN classifier... on essays--------------------------------------------
    # -------------------------------------------------------------------------------------

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(pozadVystupTrain, num_classes)
    y_test = keras.utils.to_categorical(pozadVystupTest, num_classes)

    #print("Train KERAS NN classifier... on PI_features essays")

    model_path = ('PomocneSoubNeu/NNessey_model1_trigram50tis_{ep}epochs{ba}batch.h5')
    train_model_path = model_path.format(partition=train_partition_name, ep=epochs, ba=batch_size)

    x_train = train_matrix_essays
    x_test = test_matrix_essays

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    #print('x_train shape:', x_train.shape)
    #print(x_train.shape[0], 'train samples')
    #print(x_test.shape[0], 'test samples')

    if trainNN[0] == 1:
        model1 = Sequential()  # uz neni jina moznost modelu
        model1.add(Dense(units=un, activation='tanh', input_dim=x_train.shape[1], name='dense_1'))
        model1.add(Dropout(dropout, name='dropout_1'))
        #model1.add(Dense(units=un2, activation='tanh', name='dense_2'))
        #model1.add(Dropout(dropout2, name='dropout_2'))
        model1.add(Dense(num_classes, activation='softmax', name='dense_5'))

        sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model1.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])  # nstaveni ucici algoritmus

        #reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100, min_lr=0.000001)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=10, verbose=0, mode='auto', cooldown=0, min_lr=0.001)

        history1 = model1.fit(x_train, y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=1,
                              validation_data=(x_test, y_test), callbacks=[reduce_lr])  # natrenuj  .. v priade nevejde do mpameti ...  misto fit train_on_batch (nutne zabespecit nastaveni trenovani)

        score1 = model1.evaluate(x_test, y_test, verbose=0)  # vypocitej    print('Test loss:', score1[0])
        #print('Test accuracy:', score1[1])
        accKer.append(score1[1])

        model1.save(train_model_path)  # creates a HDF5 file 'my_model.h5'

        
        # list all data in history
        #print(history1.history.keys())
        # summarize history for accuracy
        plt.plot(history1.history['acc'])
        plt.plot(history1.history['val_acc'])
        plt.title('model3 accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        # summarize history for loss
        plt.figure()
        plt.plot(history1.history['loss'])
        plt.plot(history1.history['val_loss'])
        plt.title('model1 loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        

    else:
        model1 = keras.models.load_model(train_model_path)
        model1.summary()

    # ------------------------------------------------------------------------------------------
    # test  ----------------------------------------------------------------#------------------------------------------------------------------------------------------

    if testNN[0] == 1:
        predicted_probs = model1.predict(x_test)
        predicted = np.argmax(predicted_probs, axis=1)

        accNeu = evaluateResultsAcc2(soubAslozkyTest, predicted, soubTest)
        accMoje.append(accNeu)
        print('Výsledek NEU: ' + str(accNeu))
        #display_classification_results(pozadVystupTest, predicted)

        clf = svm.LinearSVC()
        clf.fit(tfidfTrain, pozadVystupTrain)
        vysledekTFIDF = clf.predict(tfidfTest)
        acc = evaluateResultsAcc2(soubAslozkyTest, vysledekTFIDF, soubTest)
        accSVM.append(acc)
        print('Výsledek SVM: ' + str(acc))

        #display_classification_results(pozadVystupTest, vysledekTFIDF)
    '''

'''
    # -------------------------------------------------------------------------------------
    # -----------------SGD classifier... on essays--------------------------------------------
    # -------------------------------------------------------------------------------------
    model_path = ('PomocneSoubNeu/SGDessays.h5')
    train_model_path = model_path.format(partition=train_partition_name)
    if trainNN[1] == 1:
        print("Training the SGDClassifier...")
        clsf = SGDClassifier(loss='modified_huber', penalty='elasticnet', n_iter=40, n_jobs=5, alpha=1e-4)
        clsf.fit(train_matrix_essays, pozadVystupTrain)
        pickle.dump(clsf, open(train_model_path, 'wb'))
    else:
        clsf = pickle.load(open(train_model_path, 'rb'))

    if testNN[1] == 1:
        predicted = clsf.predict(test_matrix_essays)
        acc = evaluateResultsAcc2(soubAslozkyTest, predicted, soubTest)
        print(acc)

        display_classification_results(pozadVystupTest, predicted)
'''

print(vstup)
print('Výsledky po provedení ' + str(pocCrossValid) + '-fold cross validace:')
print('Výsledky s využitím TF-IDF matice.')
print('SVM: ' + str(sum(accSVM) / float(pocCrossValid)))
print('NeuKer: ' + str(sum(accKer) / float(pocCrossValid)))
print('Neu: ' + str(sum(accMoje) / float(pocCrossValid)))
print()
print('Výsledky s využitím TF-IDF matice snížené s LSA.')
print('SVM: ' + str(sum(accSVMLSA) / float(pocCrossValid)))
print('NeuKer: ' + str(sum(accKerLSA) / float(pocCrossValid)))
print('Neu: ' + str(sum(accMojeLSA) / float(pocCrossValid)))
print()
print('Výsledky s využitím doc2vec matice.')
print('SVM: ' + str(sum(accSVMoc2vec) / float(pocCrossValid)))
print('NeuKer: ' + str(sum(accKerdoc2vec) / float(pocCrossValid)))
print('Neu: ' + str(sum(accMojeoc2vec) / float(pocCrossValid)))
print()
print('Výsledky s využitím doc2vec matice snížené s LSA.')
print('SVM: ' + str(sum(accSVMdoc2vecLSA) / float(pocCrossValid)))
print('NeuKer: ' + str(sum(accKerdoc2vecLSA) / float(pocCrossValid)))
print('Neu: ' + str(sum(accMojedoc2vecLSA) / float(pocCrossValid)))
print()
print('Výsledky s využitím spojení TF-IDF snížené s LSA a doc2vec snížené s LSA matice.')
print('SVM: ' + str(sum(accSVMtfidfdoc2vec) / float(pocCrossValid)))
print('NeuKer: ' + str(sum(accKertfidfdoc2vec) / float(pocCrossValid)))
print('Neu: ' + str(sum(accMojetfidfdoc2vec) / float(pocCrossValid)))
print()
print('Výsledky s využitím spojení TF-IDF a doc2vec matice.')
print('SVM: ' + str(sum(accSVMCelTFIDFaD2V) / float(pocCrossValid)))
print('NeuKer: ' + str(sum(accKerCelTFIDFaD2V) / float(pocCrossValid)))
print('Neu: ' + str(sum(accMojeCelTFIDFaD2V) / float(pocCrossValid)))

file0 = file(u'VýsledkyNEU' + vstupPrac, 'w')
file0.write(codecs.BOM_UTF8)
file0.write(u'Výsledky po provedení '.encode('utf8') + str(pocCrossValid).encode('utf8') + u'-fold cross validace:'.encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'Výsledky s využitím TF-IDF matice.'.encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'SVM: '.encode('utf8') + str(sum(accSVM) / float(pocCrossValid)).encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'NeuKer: '.encode('utf8') + str(sum(accKer) / float(pocCrossValid)).encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'Neu: '.encode('utf8') + str(sum(accMoje) / float(pocCrossValid)).encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'Výsledky s využitím TF-IDF matice snížené s LSA.'.encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'SVM: '.encode('utf8') + str(sum(accSVMLSA) / float(pocCrossValid)).encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'NeuKer: '.encode('utf8') + str(sum(accKerLSA) / float(pocCrossValid)).encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'Neu: '.encode('utf8') + str(sum(accMojeLSA) / float(pocCrossValid)).encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'Výsledky s využitím doc2vec matice.'.encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'SVM: '.encode('utf8') + str(sum(accSVMoc2vec) / float(pocCrossValid)).encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'NeuKer: '.encode('utf8') + str(sum(accKerdoc2vec) / float(pocCrossValid)).encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'Neu: '.encode('utf8') + str(sum(accMojeoc2vec) / float(pocCrossValid)).encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'Výsledky s využitím doc2vec matice snížené s LSA.'.encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'SVM: '.encode('utf8') + str(sum(accSVMdoc2vecLSA) / float(pocCrossValid)).encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'NeuKer: '.encode('utf8') + str(sum(accKerdoc2vecLSA) / float(pocCrossValid)).encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'Neu: '.encode('utf8') + str(sum(accMojedoc2vecLSA) / float(pocCrossValid)).encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'Výsledky s využitím spojení TF-IDF LSA a doc2vec LSA matice.'.encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'SVM: '.encode('utf8') + str(sum(accSVMtfidfdoc2vec) / float(pocCrossValid)).encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'NeuKer: '.encode('utf8') + str(sum(accKertfidfdoc2vec) / float(pocCrossValid)).encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'Neu: '.encode('utf8') + str(sum(accMojetfidfdoc2vec) / float(pocCrossValid)).encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'Výsledky s využitím spojení TF-IDF a doc2vec matice.'.encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'SVM: '.encode('utf8') + str(sum(accSVMCelTFIDFaD2V) / float(pocCrossValid)).encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'NeuKer: '.encode('utf8') + str(sum(accKerCelTFIDFaD2V) / float(pocCrossValid)).encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.write(u'Neu: '.encode('utf8') + str(sum(accMojeCelTFIDFaD2V) / float(pocCrossValid)).encode('utf8'))
file0.write(u'\n'.encode('utf8'))
file0.close()