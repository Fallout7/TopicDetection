# -*- coding: utf-8 -*-
# coding: utf-8

from predpriprava import *
from kmeansMethods import UdelejKmeans, UdelejKmeansRefinementAlg
import theano
import numpy as np
from evaluation import evaluateResults
import scipy, h5py
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.layers.recurrent import LSTM
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from evaluation import *


def LSTMneu(x_train0, y_train0, x_test0, pocShluku, nazNeu):
    # ---------------------------- supervised neu ----------------------------------------------------
    Y = np.array(y_train0)
    if type(x_train0) == list:
        x_train = np.array(x_train0)
    else:
        x_train = x_train0
    y_train = keras.utils.to_categorical(Y, num_classes=pocShluku)
    if type(x_test0) == list:
        x_test = np.array(x_test0)
    else:
        x_test = x_test0
    hlSb = nazNeu + '.m'
    souboryPS, slozkyPS = ZiskejNazvySouboru('PomocneSoubNeu/', hlSb)
    if len(souboryPS) == 0:
        m = Sequential()
        m.add(Dense(640, activation='relu', input_dim=x_train.shape[1]))
        m.add(Dropout(0.5))
        m.add(Dense(2560, activation='relu'))
        m.add(Dropout(0.5))
        m.add(Dense(3624, activation='relu'))
        m.add(Dropout(0.5))
        m.add(Dense(1028, activation='relu'))
        m.add(Dropout(0.5))
        m.add(Dense(pocShluku, activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        m.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        m.fit(x_train, y_train, epochs=10000, batch_size=128)
        m.save('PomocneSoubNeu/' + hlSb)
    else:
        m = load_model('PomocneSoubNeu/' + hlSb)
    vys = m.predict(x_test, batch_size=128)

    vys = np.argmax(vys, axis=-1)

    return vys

# provede shlukování, dát si pozor aby vstup byl složka + jaká data se používají
def LSTMclustering(tfidfMat, maticeDoc2VecVah, maticeWord2VecVah, soubAslozky, nazvySoub, Ncomponents, vstupPrac):
    file0 = file(u'VýsledkyLSTM' + vstupPrac, 'w')
    file0.write(codecs.BOM_UTF8)

    pocShluku = 0
    uzJeSl = {}
    for keyy in soubAslozky:
        if not uzJeSl.has_key(soubAslozky[keyy]):
            uzJeSl[soubAslozky[keyy]] = soubAslozky[keyy]
            pocShluku += 1
    if pocShluku == 1:
        pocShluku = 31
    print 'Počet shluků je nastaven na: ' + str(pocShluku)

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

    # vytvoření tfidf matice se sníženou dimenzí za pomoci LSA
    svd = TruncatedSVD(algorithm='randomized', n_components=Ncomponents, n_iter=10, random_state=None,
                       tol=0.0)

    normalizer = Normalizer(norm='l2', copy=False)

    lsa = make_pipeline(svd, normalizer)
    tfidfMatLSA = lsa.fit_transform(tfidfMat)

    # vytvoření doc2vec matice se sníženou dimenzí za pomoci LSA
    svd = TruncatedSVD(algorithm='randomized', n_components=Ncomponents, n_iter=10, random_state=None,
                       tol=0.0)

    normalizer = Normalizer(norm='l2', copy=False)

    lsa = make_pipeline(svd, normalizer)
    maticeDoc2VecVahLSA = lsa.fit_transform(maticeDoc2VecVah)

    maticeTFIDFD2V = np.append(tfidfMatLSA, maticeDoc2VecVahLSA, axis=1)

    # --------------------------------------------  bez cross validace ------------------------------------------------
    file0.write(u'Výsledky LSTM bez cross validace vstupu: '.encode('utf8') + vstupPrac.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    # tfidf
    vysledekTFIDF = LSTMneu(tfidfMat, pozadVystup, tfidfMat, pocShluku, 'TFIDFbcrval')

    Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, nazvySoub)
    vysShl = []
    for vys in vysledekTFIDF:
        vysShl.append(cisloToShl[vys])

    acc = evaluateResultsAcc(soubAslozky, vysShl, nazvySoub)
    print 'LSTM s TFIDF matici -- Accuracy: ' + str(acc) + '; Precision: ' + str(Pp) + '; Recall: ' + str(Rr)
    file0.write(u'---------- LSTM neu na TFIDF matici ----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Accuracy je: '.encode('utf8') + str(acc).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Precision je: '.encode('utf8') + str(Pp).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Recall je: '.encode('utf8') + str(Rr).encode('utf8'))
    file0.write('\n'.encode('utf8'))

    # tfidf s LSA
    vysledekTFIDF = LSTMneu(tfidfMatLSA, pozadVystup, tfidfMatLSA, pocShluku, 'TFIDFLSAbcrval')

    Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, nazvySoub)
    vysShl = []
    for vys in vysledekTFIDF:
        vysShl.append(cisloToShl[vys])

    acc = evaluateResultsAcc(soubAslozky, vysShl, nazvySoub)
    print 'LSTM neu s TFIDF s LSA matici -- Accuracy: ' + str(acc) + '; Precision: ' + str(Pp) + '; Recall: ' + str(Rr)
    file0.write(u'---------- LSTM neu s TFIDF matici se sníženou dimenzí za použití LSA na '.encode('utf8') + str(
        Ncomponents).encode('utf8') + u'----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))

    file0.write(u'Accuracy je: '.encode('utf8') + str(acc).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Precision je: '.encode('utf8') + str(Pp).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Recall je: '.encode('utf8') + str(Rr).encode('utf8'))
    file0.write('\n'.encode('utf8'))

    # doc2vec
    vysledekTFIDF = LSTMneu(maticeDoc2VecVah, pozadVystup, maticeDoc2VecVah, pocShluku, 'doc2vecbcrval')

    Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, nazvySoub)
    vysShl = []
    for vys in vysledekTFIDF:
        vysShl.append(cisloToShl[vys])

    acc = evaluateResultsAcc(soubAslozky, vysShl, nazvySoub)
    print 'LSTM neu s doc2vec matici -- Accuracy: ' + str(acc) + '; Precision: ' + str(Pp) + '; Recall: ' + str(Rr)
    file0.write(u'---------- LSTM neu s doc2vec matici ----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Accuracy je: '.encode('utf8') + str(acc).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Precision je: '.encode('utf8') + str(Pp).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Recall je: '.encode('utf8') + str(Rr).encode('utf8'))
    file0.write('\n'.encode('utf8'))


    # doc2vec s LSA
    vysledekTFIDF = LSTMneu(maticeDoc2VecVahLSA, pozadVystup, maticeDoc2VecVahLSA, pocShluku, 'doc2vecLSAbcrval')

    Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, nazvySoub)
    vysShl = []
    for vys in vysledekTFIDF:
        vysShl.append(cisloToShl[vys])

    acc = evaluateResultsAcc(soubAslozky, vysShl, nazvySoub)
    print 'LSTM neu s doc2vec s LSA matici -- Accuracy: ' + str(acc)  # + '; Precision: ' + str(Pp) + '; Recall: ' + str(Rr)
    file0.write(
        u'---------- LSTM neu s doc2vec matici se sníženou dimenzí za použití LSA na '.encode('utf8') + str(
            Ncomponents).encode(
            'utf8') + u'----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))

    file0.write(u'Accuracy je: '.encode('utf8') + str(acc).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Precision je: '.encode('utf8') + str(Pp).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Recall je: '.encode('utf8') + str(Rr).encode('utf8'))
    file0.write('\n'.encode('utf8'))


    # tfidf a doc2vec z LSA matice spojené
    vysledekTFIDF = LSTMneu(maticeTFIDFD2V, pozadVystup, maticeTFIDFD2V, pocShluku, 'TFIDFdoc2vecbcrval')

    Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, nazvySoub)
    vysShl = []
    for vys in vysledekTFIDF:
        vysShl.append(cisloToShl[vys])

    acc = evaluateResultsAcc(soubAslozky, vysShl, nazvySoub)
    print 'LSTM neu s maticí získanou spojením tfidf matice a doc2vec matice -- Accuracy: ' + str(
        acc)  # + '; Precision: ' + str(Pp) + '; Recall: ' + str(Rr)
    file0.write(
        u'---------- LSTM neu s tfidf matici spojenou s doc2vec maticí s vektorem příznaků o velikosti '.encode(
            'utf8') + str(
            2 * Ncomponents).encode(
            'utf8') + u'----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))

    file0.write(u'Accuracy je: '.encode('utf8') + str(acc).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Precision je: '.encode('utf8') + str(Pp).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Recall je: '.encode('utf8') + str(Rr).encode('utf8'))

    file0.write('\n'.encode('utf8'))
    file0.write('\n'.encode('utf8'))
    # -------------------------------------- s cross validation -----------------------------------------------------------
    print
    pocSoub = len(soubAslozky)
    pocCrossValid = 10
    pocTrain = int((pocSoub / 100.0) * 75.0)
    pocTest = pocSoub - pocTrain
    skokPozZacTrain = int(pocSoub / float(pocCrossValid))
    print 'Probíhá ' + str(pocCrossValid) + '-fold cross validace.'
    file0.write(
        u'Výsledky LSTM neu s '.encode('utf8') + str(pocCrossValid).encode('utf8') + u'-fold cross validací vstupu: '.encode(
            'utf8') + vstupPrac.encode(
            'utf8'))
    file0.write(u'\n'.encode('utf8'))

    Acctfidf, AcctfidfLSA, Accdoc2vec, Accdoc2vecLSA, Acctfidfdoc2vec, Accword2vec, Accword2vecLSA = [], [], [], [], [], [], []
    Ptfidf, PtfidfLSA, Pdoc2vec, Pdoc2vecLSA, Rtfidf, RtfidfLSA, Rdoc2vec, Rdoc2vecLSA, PTFIDFdoc2vecLSA, RTFIDFdoc2vecLSA = [], [], [], [], [], [], [], [], [], []

    tfidfMat = tfidfMat.toarray()

    for crossValid in range(pocCrossValid):
        pocPozTr = crossValid * skokPozZacTrain
        pocPozTs = pocPozTr + pocTrain
        print 'Cyklus: ' + str(crossValid)
        if pocPozTs <= pocSoub:
            soubTrain = nazvySoub[pocPozTr:pocPozTs]
            pozadVystupTrain = pozadVystup[pocPozTr:pocPozTs]
            tfidfTrain = tfidfMat[pocPozTr:pocPozTs]
            tfidfLSATrain = tfidfMatLSA[pocPozTr:pocPozTs]
            doc2vecTrain = maticeDoc2VecVah[pocPozTr:pocPozTs]
            doc2vecLSATrain = maticeDoc2VecVahLSA[pocPozTr:pocPozTs]
            maticeTFIDFD2Vtrain = maticeTFIDFD2V[pocPozTr:pocPozTs]
            word2vecTrain = maticeWord2VecVah[pocPozTr:pocPozTs]
            # word2vecLSATrain = maticeWord2VecVahLSA[pocPozTr:pocPozTs]
        else:
            konTr = pocTrain - len(nazvySoub[pocPozTr:pocSoub - 1])
            soubTrain = nazvySoub[pocPozTr:pocSoub - 1] + nazvySoub[0:konTr]
            pozadVystupTrain = pozadVystup[pocPozTr:pocSoub - 1] + pozadVystup[0:konTr]
            tfidfTrain = np.append(tfidfMat[pocPozTr:pocSoub - 1], tfidfMat[0:konTr], axis=0)
            tfidfLSATrain = np.append(tfidfMatLSA[pocPozTr:pocSoub - 1], tfidfMatLSA[0:konTr], axis=0)
            doc2vecTrain = maticeDoc2VecVah[pocPozTr:pocSoub - 1] + maticeDoc2VecVah[0:konTr]
            doc2vecLSATrain = np.append(maticeDoc2VecVahLSA[pocPozTr:pocSoub - 1], maticeDoc2VecVahLSA[0:konTr], axis=0)
            maticeTFIDFD2Vtrain = np.append(maticeTFIDFD2V[pocPozTr:pocSoub - 1], maticeTFIDFD2V[0:konTr], axis=0)
            word2vecTrain = maticeWord2VecVah[pocPozTr:pocSoub - 1] + maticeWord2VecVah[0:konTr]
            # word2vecLSATrain = np.append(maticeWord2VecVahLSA[pocPozTr:pocSoub - 1], maticeWord2VecVahLSA[0:konTr], axis=0)

        if pocPozTs + pocTest <= pocSoub:
            soubTest = nazvySoub[pocPozTs:pocPozTs + pocTest]
            tfidfTest = tfidfMat[pocPozTs:pocPozTs + pocTest]
            tfidfLSATest = tfidfMatLSA[pocPozTs:pocPozTs + pocTest]
            doc2vecTest = maticeDoc2VecVah[pocPozTs:pocPozTs + pocTest]
            doc2vecLSATest = maticeDoc2VecVahLSA[pocPozTs:pocPozTs + pocTest]
            maticeTFIDFD2VTest = maticeTFIDFD2V[pocPozTs:pocPozTs + pocTest]
            word2vecTest = maticeWord2VecVah[pocPozTs:pocPozTs + pocTest]
            # word2vecLSATest = maticeWord2VecVahLSA[pocPozTs:pocPozTs + pocTest]
        else:
            konTs = pocTest - len(nazvySoub[pocPozTs:pocSoub - 1])
            soubTest = nazvySoub[pocPozTs:pocSoub - 1] + nazvySoub[0:konTs]
            tfidfTest = np.append(tfidfMat[pocPozTs:pocSoub - 1], tfidfMat[0:konTs], axis=0)
            tfidfLSATest = np.append(tfidfMatLSA[pocPozTs:pocSoub - 1], tfidfMatLSA[0:konTs], axis=0)
            doc2vecTest = maticeDoc2VecVah[pocPozTs:pocSoub - 1] + maticeDoc2VecVah[0:konTs]
            doc2vecLSATest = np.append(maticeDoc2VecVahLSA[pocPozTs:pocSoub - 1], maticeDoc2VecVahLSA[0:konTs], axis=0)
            maticeTFIDFD2VTest = np.append(maticeTFIDFD2V[pocPozTs:pocSoub - 1], maticeTFIDFD2V[0:konTs], axis=0)
            word2vecTest = maticeWord2VecVah[pocPozTs:pocSoub - 1] + maticeWord2VecVah[0:konTs]
            # word2vecLSATest = np.append(maticeWord2VecVahLSA[pocPozTs:pocSoub - 1], maticeWord2VecVahLSA[0:konTs], axis=0)

        soubAslozkyTest = {}
        for soub in soubTest:
            soubAslozkyTest[soub] = soubAslozky[soub]
        # tfidf
        vysledekTFIDF = LSTMneu(tfidfTrain, pozadVystupTrain, tfidfTest, pocShluku, 'TFIDFscrval')

        vysShl = []
        for vys in vysledekTFIDF:
            vysShl.append(cisloToShl[vys])
        acc = evaluateResultsAcc(soubAslozkyTest, vysShl, soubTest)


        #Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, soubTest)
        Pp, Rr = 0.0, 0.0
        Ptfidf.append(Pp)
        Rtfidf.append(Rr)
        Acctfidf.append(acc)

        # tfidf s LSA
        vysledekTFIDF = LSTMneu(tfidfLSATrain, pozadVystupTrain, tfidfLSATest, pocShluku, 'TFIDFLSAscrval')
        vysShl = []
        for vys in vysledekTFIDF:
            vysShl.append(cisloToShl[vys])
        acc = evaluateResultsAcc(soubAslozkyTest, vysShl, soubTest)
        # Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, soubTest)
        Pp, Rr = 0.0, 0.0
        PtfidfLSA.append(Pp)
        RtfidfLSA.append(Rr)
        AcctfidfLSA.append(acc)

        # doc2vec
        vysledekTFIDF = LSTMneu(doc2vecTrain, pozadVystupTrain, doc2vecTest, pocShluku, 'doc2vecscrval')

        vysShl = []
        for vys in vysledekTFIDF:
            vysShl.append(cisloToShl[vys])
        acc = evaluateResultsAcc(soubAslozkyTest, vysShl, soubTest)
        # Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, soubTest)
        Pp, Rr = 0.0, 0.0
        Pdoc2vec.append(Pp)
        Rdoc2vec.append(Rr)
        Accdoc2vec.append(acc)

        # doc2vec s LSA
        vysledekTFIDF = LSTMneu(doc2vecLSATrain, pozadVystupTrain, doc2vecLSATest, pocShluku, 'doc2vecLSAscrval')

        vysShl = []
        for vys in vysledekTFIDF:
            vysShl.append(cisloToShl[vys])
        acc = evaluateResultsAcc(soubAslozkyTest, vysShl, soubTest)
        # Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, soubTest)
        Pp, Rr = 0.0, 0.0
        Pdoc2vecLSA.append(Pp)
        Rdoc2vecLSA.append(Rr)
        Accdoc2vecLSA.append(acc)

        # tfidf spojená s doc2vec
        vysledekTFIDF = LSTMneu(maticeTFIDFD2Vtrain, pozadVystupTrain, maticeTFIDFD2VTest, pocShluku, 'TFIDFdoc2vecscrval')
        vysShl = []
        for vys in vysledekTFIDF:
            vysShl.append(cisloToShl[vys])
        acc = evaluateResultsAcc(soubAslozkyTest, vysShl, soubTest)
        # Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, soubTest)
        Pp, Rr = 0.0, 0.0
        PTFIDFdoc2vecLSA.append(Pp)
        RTFIDFdoc2vecLSA.append(Rr)
        Acctfidfdoc2vec.append(acc)

    acctfidf = sum(Acctfidf) / float(pocCrossValid)
    acctfidfLSA = sum(AcctfidfLSA) / float(pocCrossValid)
    # accword2vec = sum(Accword2vec) / float(pocCrossValid)
    # accword2vecLSA = sum(Accword2vecLSA) / float(pocCrossValid)
    accdoc2vec = sum(Accdoc2vec) / float(pocCrossValid)
    accdoc2vecLSA = sum(Accdoc2vecLSA) / float(pocCrossValid)
    acctfidfdoc2vec = sum(Acctfidfdoc2vec) / float(pocCrossValid)

    Pptfidf = sum(PtfidfLSA) / float(pocCrossValid)
    PptfidfLSA = sum(Ptfidf) / float(pocCrossValid)
    Ppdoc2vec = sum(Pdoc2vec) / float(pocCrossValid)
    Ppdoc2vecLSA = sum(Pdoc2vecLSA) / float(pocCrossValid)
    Ptfidfdoc2vec = sum(PTFIDFdoc2vecLSA) / float(pocCrossValid)

    Rrtfidf = sum(Rtfidf) / float(pocCrossValid)
    RrtfidfLSA = sum(RtfidfLSA) / float(pocCrossValid)
    Rrdoc2vec = sum(Rdoc2vec) / float(pocCrossValid)
    Rrdoc2vecLSA = sum(Rdoc2vecLSA) / float(pocCrossValid)
    Rtfidfdoc2vec = sum(RTFIDFdoc2vecLSA) / float(pocCrossValid)

    print 'LSTM neu s TFIDF matici -- Accuracy: ' + str(
        acctfidf) + '; Precision: ' + str(Pptfidf) + '; Recall: ' + str(Rrtfidf)
    print 'LSTM neu s TFIDF s LSA matici -- Accuracy: ' + str(
        acctfidfLSA) + '; Precision: ' + str(PptfidfLSA) + '; Recall: ' + str(RrtfidfLSA)

    print 'LSTM neu s doc2vec matici -- Accuracy: ' + str(
        accdoc2vec) + '; Precision: ' + str(Ppdoc2vec) + '; Recall: ' + str(Rrdoc2vec)
    print 'LSTM neu s doc2vec s LSA matici -- Accuracy: ' + str(
        accdoc2vecLSA) + '; Precision: ' + str(Ppdoc2vecLSA) + '; Recall: ' + str(Rrdoc2vecLSA)
    print 'LSTM neu s maticí tvořenou spojením TFIDF a doc2vec matic -- Accuracy: ' + str(
        acctfidfdoc2vec) + '; Precision: ' + str(Ptfidfdoc2vec) + '; Recall: ' + str(Rtfidfdoc2vec)

    file0.write(u'---------- LSTM neu s TFIDF matici ----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Accuracy je: '.encode('utf8') + str(acctfidf).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Precision je: '.encode('utf8') + str(Pptfidf).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Recall je: '.encode('utf8') + str(Rrtfidf).encode('utf8'))
    file0.write('\n'.encode('utf8'))

    file0.write(u'---------- LSTM neu s TFIDF matici se sníženou dimenzí za použití LSA na '.encode('utf8') + str(
        Ncomponents).encode('utf8') + u'----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Accuracy je: '.encode('utf8') + str(acctfidfLSA).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Precision je: '.encode('utf8') + str(PptfidfLSA).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Recall je: '.encode('utf8') + str(RrtfidfLSA).encode('utf8'))
    file0.write('\n'.encode('utf8'))

    file0.write(u'---------- LSTM neu s doc2vec matici ----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Accuracy je: '.encode('utf8') + str(accdoc2vec).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Precision je: '.encode('utf8') + str(Ppdoc2vec).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Recall je: '.encode('utf8') + str(Rrdoc2vec).encode('utf8'))
    file0.write('\n'.encode('utf8'))


    file0.write(u'---------- LSTM neu s doc2vec matici se sníženou dimenzí za použití LSA na '.encode('utf8') + str(
            Ncomponents).encode(
            'utf8') + u'----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Accuracy je: '.encode('utf8') + str(accdoc2vecLSA).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Precision je: '.encode('utf8') + str(Ppdoc2vecLSA).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Recall je: '.encode('utf8') + str(Rrdoc2vecLSA).encode('utf8'))

    file0.write(u'---------- LSTM neu s maticí tvořenou spojením tfidf matice a doc2vec matice s dimenzí '.encode('utf8') + str(
            2 * Ncomponents).encode(
            'utf8') + u'----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Accuracy je: '.encode('utf8') + str(acctfidfdoc2vec).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Precision je: '.encode('utf8') + str(Ptfidfdoc2vec).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Recall je: '.encode('utf8') + str(Rtfidfdoc2vec).encode('utf8'))

    file0.close()




