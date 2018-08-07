# -*- coding: utf-8 -*-
# coding: utf-8

import time
from sklearn import svm
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from evaluation import *
import numpy as np
from statistika import *


# provede shlukování, dát si pozor aby vstup byl složka + jaká data se používají
def UdelejSVMTRAINTEST2(tfidfMat, textyTrain, soubAslozky, nazvySoubTrain, Ncomponents, vstupPrac2, PaR, slovnik, slovnikPole):
    file0 = file(u'VýsledkySVMTRAINTEST2' + vstupPrac2, 'w')
    file0.write(codecs.BOM_UTF8)
    print "Jo jede to rozdělený na ty trénovací a testovací data."
    pozadVystup = []
    propojeni = {}
    cisloToShl = {}
    cisloShl = 0
    for soub in nazvySoubTrain:
        if not propojeni.has_key(soubAslozky[soub]):
            pozadVystup.append(cisloShl)
            propojeni[soubAslozky[soub]] = cisloShl
            cisloToShl[cisloShl] = soubAslozky[soub]
            cisloShl += 1
        else:
            pozadVystup.append(propojeni[soubAslozky[soub]])


    file0.write(u'Výsledky SVM vstupu: '.encode('utf8') + vstupPrac2.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))

    vstup = 'Vstup3rawTest'
    soubAtextyRaw, soubAslozky = NacteniRawVstupu(vstup)
    vycisteneTexty, lemmaTexty, tagsTexty = UpravAVysictiTexty(soubAtextyRaw, vstup, 'english')

    vstupPrac = vstup + 'CistyText'
    textyPracovni = vycisteneTexty

    tfidfMatTest, nazvySoub = vytvorTFIDF(vstupPrac, textyPracovni, slovnikPole)

    velikost, okno, alphaa, minalphaa, minimalniPocetCetnostiSlov = 5000, 0.01, 0.019, 0.019, 5

    # vytvoření tfidf matice se sníženou dimenzí za pomoci LSA
    svd = TruncatedSVD(algorithm='randomized', n_components=Ncomponents, n_iter=10, random_state=None,
                       tol=0.0)

    normalizer = Normalizer(norm='l2', copy=False)

    lsa = make_pipeline(svd, normalizer)
    tfidfMatLSA = lsa.fit_transform(tfidfMat)

    # vytvoření  TEST tfidf matice se sníženou dimenzí za pomoci LSA
    tfidfMatLSATest = lsa.transform(tfidfMatTest)
    print tfidfMatLSATest.shape


    maticeDoc2VecVah, maticeDoc2VecVahTest = VytvorReprDoc2VecTrainTest(vstupPrac2, textyTrain, textyPracovni,
                                                                        nazvySoubTrain, nazvySoub, velikost, okno,
                                                                        alphaa,
                                                                        minalphaa, minimalniPocetCetnostiSlov)
    # vytvoření doc2vec matice se sníženou dimenzí za pomoci LSA
    svd = TruncatedSVD(algorithm='randomized', n_components=Ncomponents, n_iter=10, random_state=None,
                       tol=0.0)

    normalizer = Normalizer(norm='l2', copy=False)

    lsa = make_pipeline(svd, normalizer)
    maticeDoc2VecVahLSA = lsa.fit_transform(maticeDoc2VecVah)

    maticeTFIDFD2V = np.append(tfidfMatLSA, maticeDoc2VecVahLSA, axis=1)

    # vytvoření TEST doc2vec matice se sníženou dimenzí za pomoci LSA
    maticeDoc2VecVahLSATest = lsa.transform(maticeDoc2VecVahTest)

    maticeTFIDFD2VTest = np.append(tfidfMatLSATest, maticeDoc2VecVahLSATest, axis=1)


    #tfidf
    clf = svm.LinearSVC()
    clf.fit(tfidfMat, pozadVystup)
    vysledekTFIDF = clf.predict(tfidfMatTest)

    if PaR == 1:
        Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, nazvySoub)
    vysShl = []
    for vys in vysledekTFIDF:
        vysShl.append(cisloToShl[vys])

    acc = evaluateResultsAcc(soubAslozky, vysShl, nazvySoub)
    print 'SVM na TFIDF matici -- Accuracy: ' + str(acc) #+ '; Precision: ' + str(Pp) + '; Recall: ' + str(Rr)
    file0.write(u'---------- SVM na TFIDF matici ----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Accuracy je: '.encode('utf8') + str(acc).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    if PaR == 1:
        file0.write(u'Precision je: '.encode('utf8') + str(Pp).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Recall je: '.encode('utf8') + str(Rr).encode('utf8'))
        file0.write('\n'.encode('utf8'))

    # tfidf s LSA
    clf = svm.LinearSVC()
    clf.fit(tfidfMatLSA, pozadVystup)
    vysledekTFIDF = clf.predict(tfidfMatLSATest)
    if PaR == 1:
        Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, nazvySoub)
    vysShl = []
    for vys in vysledekTFIDF:
        vysShl.append(cisloToShl[vys])

    acc = evaluateResultsAcc(soubAslozky, vysShl, nazvySoub)
    print 'SVM na TFIDF s LSA matici -- Accuracy: ' + str(acc) #+ '; Precision: ' + str(Pp) + '; Recall: ' + str(Rr)
    file0.write(u'---------- SVM na TFIDF matici se sníženou dimenzí za použití LSA na '.encode('utf8') + str(Ncomponents).encode('utf8') + u'----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))

    file0.write(u'Accuracy je: '.encode('utf8') + str(acc).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    if PaR == 1:
        file0.write(u'Precision je: '.encode('utf8') + str(Pp).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Recall je: '.encode('utf8') + str(Rr).encode('utf8'))
        file0.write('\n'.encode('utf8'))

    # doc2vec
    clf = svm.LinearSVC()
    clf.fit(maticeDoc2VecVah, pozadVystup)
    vysledekTFIDF = clf.predict(maticeDoc2VecVahTest)
    if PaR == 1:
        Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, nazvySoub)
    vysShl = []
    for vys in vysledekTFIDF:
        vysShl.append(cisloToShl[vys])

    acc = evaluateResultsAcc(soubAslozky, vysShl, nazvySoub)
    print 'SVM na doc2vec matici -- Accuracy: ' + str(
        acc)  # + '; Precision: ' + str(Pp) + '; Recall: ' + str(Rr)
    file0.write(u'---------- SVM na doc2vec matici ----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))

    file0.write(u'Accuracy je: '.encode('utf8') + str(acc).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    if PaR == 1:
        file0.write(u'Precision je: '.encode('utf8') + str(Pp).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Recall je: '.encode('utf8') + str(Rr).encode('utf8'))
        file0.write('\n'.encode('utf8'))

    # doc2vec s LSA
    clf = svm.LinearSVC()
    clf.fit(maticeDoc2VecVahLSA, pozadVystup)
    vysledekTFIDF = clf.predict(maticeDoc2VecVahLSATest)
    if PaR == 1:
        Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, nazvySoub)
    vysShl = []
    for vys in vysledekTFIDF:
        vysShl.append(cisloToShl[vys])

    acc = evaluateResultsAcc(soubAslozky, vysShl, nazvySoub)
    print 'SVM na doc2vec s LSA matici -- Accuracy: ' + str(
        acc)  # + '; Precision: ' + str(Pp) + '; Recall: ' + str(Rr)
    file0.write(
        u'---------- SVM na doc2vec matici se sníženou dimenzí za použití LSA na '.encode('utf8') + str(
            Ncomponents).encode(
            'utf8') + u'----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))

    file0.write(u'Accuracy je: '.encode('utf8') + str(acc).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    if PaR == 1:
        file0.write(u'Precision je: '.encode('utf8') + str(Pp).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Recall je: '.encode('utf8') + str(Rr).encode('utf8'))
        file0.write('\n'.encode('utf8'))

    # tfidf a doc2vec z LSA matice spojené
    clf = svm.LinearSVC()
    clf.fit(maticeTFIDFD2V, pozadVystup)
    vysledekTFIDF = clf.predict(maticeTFIDFD2VTest)
    if PaR == 1:
        Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, nazvySoub)
    vysShl = []
    for vys in vysledekTFIDF:
        vysShl.append(cisloToShl[vys])

    acc = evaluateResultsAcc(soubAslozky, vysShl, nazvySoub)
    print 'SVM s maticí získanou spojením tfidf matice a doc2vec matice -- Accuracy: ' + str(
        acc)  # + '; Precision: ' + str(Pp) + '; Recall: ' + str(Rr)
    file0.write(
        u'---------- SVM na tfidf matici spojenou s doc2vec maticí s vektorem příznaků o velikosti '.encode(
            'utf8') + str(
            2 * Ncomponents).encode(
            'utf8') + u'----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))

    file0.write(u'Accuracy je: '.encode('utf8') + str(acc).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    if PaR == 1:
        file0.write(u'Precision je: '.encode('utf8') + str(Pp).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Recall je: '.encode('utf8') + str(Rr).encode('utf8'))
        file0.write('\n'.encode('utf8'))
    file0.write('\n'.encode('utf8'))
    file0.write('\n'.encode('utf8'))
    file0.close()

# provede shlukování, dát si pozor aby vstup byl složka + jaká data se používají
def UdelejSVMTRAINTEST(tfidfMat, textyTrain, soubAslozky, nazvySoubTrain, Ncomponents, vstupPrac2, PaR, slovnik, slovnikPole):
    file0 = file(u'VýsledkySVMTRAINTEST' + vstupPrac2, 'w')
    file0.write(codecs.BOM_UTF8)
    print "Jo jede to rozdělený na ty trénovací a testovací data."
    pozadVystup = []
    propojeni = {}
    cisloToShl = {}
    cisloShl = 0
    for soub in nazvySoubTrain:
        if not propojeni.has_key(soubAslozky[soub]):
            pozadVystup.append(cisloShl)
            propojeni[soubAslozky[soub]] = cisloShl
            cisloToShl[cisloShl] = soubAslozky[soub]
            cisloShl += 1
        else:
            pozadVystup.append(propojeni[soubAslozky[soub]])

    file0.write(u'Výsledky SVM vstupu: '.encode('utf8') + vstupPrac2.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    TestVelikosti = [50, 200, 350]
    for velikostTest in TestVelikosti:
        if velikostTest == 50:
            Ncomponents = 50
        else:
            Ncomponents = 200
        print 'Počet komponent: ' + str(Ncomponents)
        print velikostTest

        vstup = 'Vstup3rawPorovTest'+ str(velikostTest)
        soubAtextyRaw, soubAslozky = NacteniRawVstupu(vstup)
        vycisteneTexty, lemmaTexty, tagsTexty = UpravAVysictiTexty(soubAtextyRaw, vstup, 'english')

        vstupPrac = vstup + 'CistyText'
        textyPracovni = vycisteneTexty

        tfidfMatTest, nazvySoub = vytvorTFIDF(vstupPrac, textyPracovni, slovnikPole)

        # vytvoření tfidf matice se sníženou dimenzí za pomoci LSA
        svd = TruncatedSVD(algorithm='randomized', n_components=Ncomponents, n_iter=10, random_state=None,
                           tol=0.0)

        normalizer = Normalizer(norm='l2', copy=False)

        lsa = make_pipeline(svd, normalizer)
        tfidfMatLSA = lsa.fit_transform(tfidfMat)

        # vytvoření  TEST tfidf matice se sníženou dimenzí za pomoci LSA
        tfidfMatLSATest = lsa.transform(tfidfMatTest)

        velikost, okno, alphaa, minalphaa, minimalniPocetCetnostiSlov = 5000, 0.01, 0.019, 0.019, 5

        maticeDoc2VecVah, maticeDoc2VecVahTest = VytvorReprDoc2VecTrainTest(vstupPrac, textyTrain, textyPracovni, nazvySoubTrain, nazvySoub, velikost, okno, alphaa,
                                                     minalphaa, minimalniPocetCetnostiSlov)
        # vytvoření doc2vec matice se sníženou dimenzí za pomoci LSA
        svd = TruncatedSVD(algorithm='randomized', n_components=Ncomponents, n_iter=10, random_state=None,
                           tol=0.0)

        normalizer = Normalizer(norm='l2', copy=False)

        lsa = make_pipeline(svd, normalizer)
        maticeDoc2VecVahLSA = lsa.fit_transform(maticeDoc2VecVah)

        # vytvoření TEST doc2vec matice se sníženou dimenzí za pomoci LSA
        maticeDoc2VecVahLSATest = lsa.transform(maticeDoc2VecVahTest)

        maticeTFIDFD2V = np.append(tfidfMatLSA, maticeDoc2VecVahLSA, axis=1)

        maticeTFIDFD2VTest = np.append(tfidfMatLSATest, maticeDoc2VecVahLSATest, axis=1)


        file0.write(u'Velikost testovacích dat: '.encode('utf8') + str(velikostTest).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        #tfidf
        clf = svm.LinearSVC()
        clf.fit(tfidfMat, pozadVystup)
        vysledekTFIDF = clf.predict(tfidfMatTest)

        if PaR == 1:
            Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, nazvySoub)
        vysShl = []
        for vys in vysledekTFIDF:
            vysShl.append(cisloToShl[vys])

        acc = evaluateResultsAcc(soubAslozky, vysShl, nazvySoub)
        print 'SVM na TFIDF matici -- Accuracy: ' + str(acc) #+ '; Precision: ' + str(Pp) + '; Recall: ' + str(Rr)
        file0.write(u'---------- SVM na TFIDF matici ----------'.encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Accuracy je: '.encode('utf8') + str(acc).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        if PaR == 1:
            file0.write(u'Precision je: '.encode('utf8') + str(Pp).encode('utf8'))
            file0.write(u'\n'.encode('utf8'))
            file0.write(u'Recall je: '.encode('utf8') + str(Rr).encode('utf8'))
            file0.write('\n'.encode('utf8'))

        # tfidf s LSA
        clf = svm.LinearSVC()
        clf.fit(tfidfMatLSA, pozadVystup)
        vysledekTFIDF = clf.predict(tfidfMatLSATest)
        if PaR == 1:
            Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, nazvySoub)
        vysShl = []
        for vys in vysledekTFIDF:
            vysShl.append(cisloToShl[vys])

        acc = evaluateResultsAcc(soubAslozky, vysShl, nazvySoub)
        print 'SVM na TFIDF s LSA matici -- Accuracy: ' + str(acc) #+ '; Precision: ' + str(Pp) + '; Recall: ' + str(Rr)
        file0.write(u'---------- SVM na TFIDF matici se sníženou dimenzí za použití LSA na '.encode('utf8') + str(Ncomponents).encode('utf8') + u'----------'.encode('utf8'))
        file0.write(u'\n'.encode('utf8'))

        file0.write(u'Accuracy je: '.encode('utf8') + str(acc).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        if PaR == 1:
            file0.write(u'Precision je: '.encode('utf8') + str(Pp).encode('utf8'))
            file0.write(u'\n'.encode('utf8'))
            file0.write(u'Recall je: '.encode('utf8') + str(Rr).encode('utf8'))
            file0.write('\n'.encode('utf8'))

        # doc2vec
        clf = svm.LinearSVC()
        clf.fit(maticeDoc2VecVah, pozadVystup)
        vysledekTFIDF = clf.predict(maticeDoc2VecVahTest)
        if PaR == 1:
            Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, nazvySoub)
        vysShl = []
        for vys in vysledekTFIDF:
            vysShl.append(cisloToShl[vys])

        acc = evaluateResultsAcc(soubAslozky, vysShl, nazvySoub)
        print 'SVM na doc2vec matici -- Accuracy: ' + str(
            acc)  # + '; Precision: ' + str(Pp) + '; Recall: ' + str(Rr)
        file0.write(u'---------- SVM na doc2vec matici ----------'.encode('utf8'))
        file0.write(u'\n'.encode('utf8'))

        file0.write(u'Accuracy je: '.encode('utf8') + str(acc).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        if PaR == 1:
            file0.write(u'Precision je: '.encode('utf8') + str(Pp).encode('utf8'))
            file0.write(u'\n'.encode('utf8'))
            file0.write(u'Recall je: '.encode('utf8') + str(Rr).encode('utf8'))
            file0.write('\n'.encode('utf8'))

        # doc2vec s LSA
        clf = svm.LinearSVC()
        clf.fit(maticeDoc2VecVahLSA, pozadVystup)
        vysledekTFIDF = clf.predict(maticeDoc2VecVahLSATest)
        if PaR == 1:
            Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, nazvySoub)
        vysShl = []
        for vys in vysledekTFIDF:
            vysShl.append(cisloToShl[vys])

        acc = evaluateResultsAcc(soubAslozky, vysShl, nazvySoub)
        print 'SVM na doc2vec s LSA matici -- Accuracy: ' + str(
            acc)  # + '; Precision: ' + str(Pp) + '; Recall: ' + str(Rr)
        file0.write(
            u'---------- SVM na doc2vec matici se sníženou dimenzí za použití LSA na '.encode('utf8') + str(
                Ncomponents).encode(
                'utf8') + u'----------'.encode('utf8'))
        file0.write(u'\n'.encode('utf8'))

        file0.write(u'Accuracy je: '.encode('utf8') + str(acc).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        if PaR == 1:
            file0.write(u'Precision je: '.encode('utf8') + str(Pp).encode('utf8'))
            file0.write(u'\n'.encode('utf8'))
            file0.write(u'Recall je: '.encode('utf8') + str(Rr).encode('utf8'))
            file0.write('\n'.encode('utf8'))

        # tfidf a doc2vec z LSA matice spojené
        clf = svm.LinearSVC()
        clf.fit(maticeTFIDFD2V, pozadVystup)
        vysledekTFIDF = clf.predict(maticeTFIDFD2VTest)
        if PaR == 1:
            Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, nazvySoub)
        vysShl = []
        for vys in vysledekTFIDF:
            vysShl.append(cisloToShl[vys])

        acc = evaluateResultsAcc(soubAslozky, vysShl, nazvySoub)
        print 'SVM s maticí získanou spojením tfidf matice a doc2vec matice -- Accuracy: ' + str(
            acc)  # + '; Precision: ' + str(Pp) + '; Recall: ' + str(Rr)
        file0.write(
            u'---------- SVM na tfidf matici spojenou s doc2vec maticí s vektorem příznaků o velikosti '.encode(
                'utf8') + str(
                2 * Ncomponents).encode(
                'utf8') + u'----------'.encode('utf8'))
        file0.write(u'\n'.encode('utf8'))

        file0.write(u'Accuracy je: '.encode('utf8') + str(acc).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        if PaR == 1:
            file0.write(u'Precision je: '.encode('utf8') + str(Pp).encode('utf8'))
            file0.write(u'\n'.encode('utf8'))
            file0.write(u'Recall je: '.encode('utf8') + str(Rr).encode('utf8'))
            file0.write('\n'.encode('utf8'))
        file0.write('\n'.encode('utf8'))
        file0.write('\n'.encode('utf8'))
    file0.close()



# provede shlukování, dát si pozor aby vstup byl složka + jaká data se používají
def UdelejSVM(tfidfMat, maticeDoc2VecVah, maticeWord2VecVah, soubAslozky, nazvySoub, Ncomponents, vstupPrac, PaR):
    file0 = file(u'VýsledkySVM' + vstupPrac, 'w')
    file0.write(codecs.BOM_UTF8)

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

    '''
    # vytvoření word2vec matice se sníženou dimenzí za pomoci LSA
    svd = TruncatedSVD(algorithm='randomized', n_components=Ncomponents, n_iter=10, random_state=None,
                       tol=0.0)

    normalizer = Normalizer(norm='l2', copy=False)

    lsa = make_pipeline(svd, normalizer)
    maticeWord2VecVahLSA = lsa.fit_transform(maticeWord2VecVah)
    '''

# --------------------------------------------  bez cross validace ------------------------------------------------
    file0.write(u'Výsledky SVM bez cross validace vstupu: '.encode('utf8') + vstupPrac.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    #tfidf
    clf = svm.LinearSVC()
    clf.fit(tfidfMat, pozadVystup)
    vysledekTFIDF = clf.predict(tfidfMat)

    if PaR == 1:
        Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, nazvySoub)
    vysShl = []
    for vys in vysledekTFIDF:
        vysShl.append(cisloToShl[vys])

    acc = evaluateResultsAcc(soubAslozky, vysShl, nazvySoub)
    print 'SVM na TFIDF matici -- Accuracy: ' + str(acc) #+ '; Precision: ' + str(Pp) + '; Recall: ' + str(Rr)
    file0.write(u'---------- SVM na TFIDF matici ----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Accuracy je: '.encode('utf8') + str(acc).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    if PaR == 1:
        file0.write(u'Precision je: '.encode('utf8') + str(Pp).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Recall je: '.encode('utf8') + str(Rr).encode('utf8'))
        file0.write('\n'.encode('utf8'))

    # tfidf s LSA
    clf = svm.LinearSVC()
    clf.fit(tfidfMatLSA, pozadVystup)
    vysledekTFIDF = clf.predict(tfidfMatLSA)
    if PaR == 1:
        Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, nazvySoub)
    vysShl = []
    for vys in vysledekTFIDF:
        vysShl.append(cisloToShl[vys])

    acc = evaluateResultsAcc(soubAslozky, vysShl, nazvySoub)
    print 'SVM na TFIDF s LSA matici -- Accuracy: ' + str(acc) #+ '; Precision: ' + str(Pp) + '; Recall: ' + str(Rr)
    file0.write(u'---------- SVM na TFIDF matici se sníženou dimenzí za použití LSA na '.encode('utf8') + str(Ncomponents).encode('utf8') + u'----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))

    file0.write(u'Accuracy je: '.encode('utf8') + str(acc).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    if PaR == 1:
        file0.write(u'Precision je: '.encode('utf8') + str(Pp).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Recall je: '.encode('utf8') + str(Rr).encode('utf8'))
        file0.write('\n'.encode('utf8'))


    '''
    # word2vec
    clf = svm.LinearSVC()
    clf.fit(maticeWord2VecVah, pozadVystup)
    vysledekTFIDF = clf.predict(maticeWord2VecVah)

    # Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, nazvySoub)
    vysShl = []
    for vys in vysledekTFIDF:
        vysShl.append(cisloToShl[vys])

    acc = evaluateResultsAcc(soubAslozky, vysShl, nazvySoub)
    print 'SVM na word2vec matici -- Accuracy: ' + str(acc)  # + '; Precision: ' + str(Pp) + '; Recall: ' + str(Rr)
    file0.write(u'---------- SVM na word2vec matici ----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))

    file0.write(u'Accuracy je: '.encode('utf8') + str(acc).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    '''

    '''
    # word2vec s LSA
    clf = svm.LinearSVC()
    clf.fit(maticeWord2VecVahLSA, pozadVystup)
    vysledekTFIDF = clf.predict(maticeWord2VecVahLSA)

    # Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, nazvySoub)
    vysShl = []
    for vys in vysledekTFIDF:
        vysShl.append(cisloToShl[vys])

    acc = evaluateResultsAcc(soubAslozky, vysShl, nazvySoub)
    print 'SVM na word2vec matici se sníženou dimenzí s LSA -- Accuracy: ' + str(acc)  # + '; Precision: ' + str(Pp) + '; Recall: ' + str(Rr)
    file0.write(u'---------- SVM na word2vec matici se sníženou dimenzí za použití LSA na '.encode('utf8') + str(Ncomponents).encode(
            'utf8') + u'----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))

    file0.write(u'Accuracy je: '.encode('utf8') + str(acc).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    '''
    # doc2vec
    clf = svm.LinearSVC()
    clf.fit(maticeDoc2VecVah, pozadVystup)
    vysledekTFIDF = clf.predict(maticeDoc2VecVah)
    if PaR == 1:
        Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, nazvySoub)
    vysShl = []
    for vys in vysledekTFIDF:
        vysShl.append(cisloToShl[vys])

    acc = evaluateResultsAcc(soubAslozky, vysShl, nazvySoub)
    print 'SVM na doc2vec matici -- Accuracy: ' + str(acc) #+ '; Precision: ' + str(Pp) + '; Recall: ' + str(Rr)
    file0.write(u'---------- SVM na doc2vec matici ----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))

    file0.write(u'Accuracy je: '.encode('utf8') + str(acc).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    if PaR == 1:
        file0.write(u'Precision je: '.encode('utf8') + str(Pp).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Recall je: '.encode('utf8') + str(Rr).encode('utf8'))
        file0.write('\n'.encode('utf8'))

    # doc2vec s LSA
    clf = svm.LinearSVC()
    clf.fit(maticeDoc2VecVahLSA, pozadVystup)
    vysledekTFIDF = clf.predict(maticeDoc2VecVahLSA)
    if PaR == 1:
        Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, nazvySoub)
    vysShl = []
    for vys in vysledekTFIDF:
        vysShl.append(cisloToShl[vys])

    acc = evaluateResultsAcc(soubAslozky, vysShl, nazvySoub)
    print 'SVM na doc2vec s LSA matici -- Accuracy: ' + str(acc) #+ '; Precision: ' + str(Pp) + '; Recall: ' + str(Rr)
    file0.write(
        u'---------- SVM na doc2vec matici se sníženou dimenzí za použití LSA na '.encode('utf8') + str(Ncomponents).encode(
            'utf8') + u'----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))

    file0.write(u'Accuracy je: '.encode('utf8') + str(acc).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    if PaR == 1:
        file0.write(u'Precision je: '.encode('utf8') + str(Pp).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Recall je: '.encode('utf8') + str(Rr).encode('utf8'))
        file0.write('\n'.encode('utf8'))


    # tfidf a doc2vec z LSA matice spojené
    clf = svm.LinearSVC()
    clf.fit(maticeTFIDFD2V, pozadVystup)
    vysledekTFIDF = clf.predict(maticeTFIDFD2V)
    if PaR == 1:
        Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, nazvySoub)
    vysShl = []
    for vys in vysledekTFIDF:
        vysShl.append(cisloToShl[vys])

    acc = evaluateResultsAcc(soubAslozky, vysShl, nazvySoub)
    print 'SVM s maticí získanou spojením tfidf matice a doc2vec matice -- Accuracy: ' + str(acc)  # + '; Precision: ' + str(Pp) + '; Recall: ' + str(Rr)
    file0.write(
        u'---------- SVM na tfidf matici spojenou s doc2vec maticí s vektorem příznaků o velikosti '.encode('utf8') + str(
            2*Ncomponents).encode(
            'utf8') + u'----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))

    file0.write(u'Accuracy je: '.encode('utf8') + str(acc).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    if PaR == 1:
        file0.write(u'Precision je: '.encode('utf8') + str(Pp).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Recall je: '.encode('utf8') + str(Rr).encode('utf8'))
        file0.write('\n'.encode('utf8'))
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
    file0.write(u'Výsledky SVM s '.encode('utf8') + str(pocCrossValid).encode('utf8') + u'-fold cross validací vstupu: '.encode('utf8') + vstupPrac.encode(
            'utf8'))
    file0.write(u'\n'.encode('utf8'))

    Acctfidf, AcctfidfLSA, Accdoc2vec, Accdoc2vecLSA, Acctfidfdoc2vec, Accword2vec, Accword2vecLSA = [], [], [], [], [], [], []
    Ptfidf, PtfidfLSA, Pdoc2vec, Pdoc2vecLSA, Rtfidf, RtfidfLSA, Rdoc2vec, Rdoc2vecLSA, Ptfidfd2v, Rtfidfd2v = [], [], [], [], [], [], [], [], [], []

    tfidfMat = tfidfMat.toarray()

    for crossValid in range(pocCrossValid):
        pocPozTr = crossValid * skokPozZacTrain
        pocPozTs = pocPozTr+pocTrain
        print 'Cyklus: ' + str(crossValid)
        if pocPozTs <= pocSoub:
            soubTrain = nazvySoub[pocPozTr:pocPozTs]
            pozadVystupTrain = pozadVystup[pocPozTr:pocPozTs]
            tfidfTrain = tfidfMat[pocPozTr:pocPozTs]
            tfidfLSATrain = tfidfMatLSA[pocPozTr:pocPozTs]
            doc2vecTrain = maticeDoc2VecVah[pocPozTr:pocPozTs]
            doc2vecLSATrain = maticeDoc2VecVahLSA[pocPozTr:pocPozTs]
            maticeTFIDFD2Vtrain = maticeTFIDFD2V[pocPozTr:pocPozTs]
            #word2vecTrain = maticeWord2VecVah[pocPozTr:pocPozTs]
            #word2vecLSATrain = maticeWord2VecVahLSA[pocPozTr:pocPozTs]
        else:
            konTr = pocTrain - len(nazvySoub[pocPozTr:pocSoub-1])
            soubTrain = nazvySoub[pocPozTr:pocSoub-1] + nazvySoub[0:konTr]
            pozadVystupTrain = pozadVystup[pocPozTr:pocSoub-1] + pozadVystup[0:konTr]
            tfidfTrain = np.append(tfidfMat[pocPozTr:pocSoub-1], tfidfMat[0:konTr], axis=0)
            tfidfLSATrain = np.append(tfidfMatLSA[pocPozTr:pocSoub-1], tfidfMatLSA[0:konTr], axis=0)
            doc2vecTrain = maticeDoc2VecVah[pocPozTr:pocSoub-1] + maticeDoc2VecVah[0:konTr]
            doc2vecLSATrain = np.append(maticeDoc2VecVahLSA[pocPozTr:pocSoub-1], maticeDoc2VecVahLSA[0:konTr], axis=0)
            maticeTFIDFD2Vtrain = np.append(maticeTFIDFD2V[pocPozTr:pocSoub - 1], maticeTFIDFD2V[0:konTr], axis=0)
            #word2vecTrain = maticeWord2VecVah[pocPozTr:pocSoub - 1] + maticeWord2VecVah[0:konTr]
            #word2vecLSATrain = np.append(maticeWord2VecVahLSA[pocPozTr:pocSoub - 1], maticeWord2VecVahLSA[0:konTr], axis=0)

        if pocPozTs + pocTest <= pocSoub:
            soubTest = nazvySoub[pocPozTs:pocPozTs + pocTest]
            tfidfTest = tfidfMat[pocPozTs:pocPozTs + pocTest]
            tfidfLSATest = tfidfMatLSA[pocPozTs:pocPozTs + pocTest]
            doc2vecTest = maticeDoc2VecVah[pocPozTs:pocPozTs + pocTest]
            doc2vecLSATest = maticeDoc2VecVahLSA[pocPozTs:pocPozTs + pocTest]
            maticeTFIDFD2VTest = maticeTFIDFD2V[pocPozTs:pocPozTs + pocTest]
            #word2vecTest = maticeWord2VecVah[pocPozTs:pocPozTs + pocTest]
            #word2vecLSATest = maticeWord2VecVahLSA[pocPozTs:pocPozTs + pocTest]
        else:
            konTs = pocTest - len(nazvySoub[pocPozTs:pocSoub - 1])
            soubTest = nazvySoub[pocPozTs:pocSoub - 1] + nazvySoub[0:konTs]
            tfidfTest = np.append(tfidfMat[pocPozTs:pocSoub - 1], tfidfMat[0:konTs], axis=0)
            tfidfLSATest = np.append(tfidfMatLSA[pocPozTs:pocSoub - 1], tfidfMatLSA[0:konTs], axis=0)
            doc2vecTest = maticeDoc2VecVah[pocPozTs:pocSoub - 1] + maticeDoc2VecVah[0:konTs]
            doc2vecLSATest = np.append(maticeDoc2VecVahLSA[pocPozTs:pocSoub - 1], maticeDoc2VecVahLSA[0:konTs], axis=0)
            maticeTFIDFD2VTest = np.append(maticeTFIDFD2V[pocPozTs:pocSoub - 1], maticeTFIDFD2V[0:konTs], axis=0)
            #word2vecTest = maticeWord2VecVah[pocPozTs:pocSoub - 1] + maticeWord2VecVah[0:konTs]
            #word2vecLSATest = np.append(maticeWord2VecVahLSA[pocPozTs:pocSoub - 1], maticeWord2VecVahLSA[0:konTs], axis=0)

        soubAslozkyTest = {}
        for soub in soubTest:
            soubAslozkyTest[soub] = soubAslozky[soub]
        # tfidf
        clf = svm.LinearSVC()
        clf.fit(tfidfTrain, pozadVystupTrain)
        vysledekTFIDF = clf.predict(tfidfTest)

        vysShl = []
        for vys in vysledekTFIDF:
            vysShl.append(cisloToShl[vys])
        acc = evaluateResultsAcc(soubAslozkyTest, vysShl, soubTest)
        acc2 = evaluateResultsAcc2(soubAslozkyTest, vysShl, soubTest)
        print
        print acc, acc2
        print
        statCeho = vstupPrac + u'SVMTFIDFcrossval' + str(crossValid)
        statistikaVytvorenychShluku(soubAslozkyTest, vysShl, soubTest, statCeho)

        if PaR == 1:
            Pp, Rr = evaluateResults(soubAslozky, vysledekTFIDF, soubTest)
            Ptfidf.append(Pp)
            Rtfidf.append(Rr)

        Acctfidf.append(acc)

        # tfidf s LSA
        clf = svm.LinearSVC()
        clf.fit(tfidfLSATrain, pozadVystupTrain)
        vysledekTFIDF = clf.predict(tfidfLSATest)

        vysShl = []
        for vys in vysledekTFIDF:
            vysShl.append(cisloToShl[vys])
        acc = evaluateResultsAcc(soubAslozkyTest, vysShl, soubTest)
        statCeho = vstupPrac + u'SVMTFIDFLSAcrossval' + str(crossValid)
        statistikaVytvorenychShluku(soubAslozkyTest, vysShl, soubTest, statCeho)
        if PaR == 1:
            Pp, Rr = evaluateResults(soubAslozkyTest, vysledekTFIDF, soubTest)
            PtfidfLSA.append(Pp)
            RtfidfLSA.append(Rr)

        AcctfidfLSA.append(acc)

        '''
        # word2vec
        clf = svm.LinearSVC()
        clf.fit(word2vecTrain, pozadVystupTrain)
        vysledekTFIDF = clf.predict(word2vecTest)

        vysShl = []
        for vys in vysledekTFIDF:
            vysShl.append(cisloToShl[vys])
        acc = evaluateResultsAcc(soubAslozkyTest, vysShl, soubTest)
        Accword2vec.append(acc)

        # word2vec s LSA
        clf = svm.LinearSVC()
        clf.fit(word2vecLSATrain, pozadVystupTrain)
        vysledekTFIDF = clf.predict(word2vecLSATest)

        vysShl = []
        for vys in vysledekTFIDF:
            vysShl.append(cisloToShl[vys])
        acc = evaluateResultsAcc(soubAslozkyTest, vysShl, soubTest)
        Accword2vecLSA.append(acc)
        '''
        # doc2vec
        clf = svm.LinearSVC()
        clf.fit(doc2vecTrain, pozadVystupTrain)
        vysledekTFIDF = clf.predict(doc2vecTest)

        vysShl = []
        for vys in vysledekTFIDF:
            vysShl.append(cisloToShl[vys])
        acc = evaluateResultsAcc(soubAslozkyTest, vysShl, soubTest)
        statCeho = vstupPrac + u'SVMdoc2veccrossval' + str(crossValid)
        statistikaVytvorenychShluku(soubAslozkyTest, vysShl, soubTest, statCeho)
        if PaR == 1:
            Pp, Rr = evaluateResults(soubAslozkyTest, vysledekTFIDF, soubTest)
            Pdoc2vec.append(Pp)
            Rdoc2vec.append(Rr)

        Accdoc2vec.append(acc)

        # doc2vec s LSA
        clf = svm.LinearSVC()
        clf.fit(doc2vecLSATrain, pozadVystupTrain)
        vysledekTFIDF = clf.predict(doc2vecLSATest)

        vysShl = []
        for vys in vysledekTFIDF:
            vysShl.append(cisloToShl[vys])
        acc = evaluateResultsAcc(soubAslozkyTest, vysShl, soubTest)
        statCeho = vstupPrac + u'SVMdoc2vecLSAcrossval' + str(crossValid)
        statistikaVytvorenychShluku(soubAslozkyTest, vysShl, soubTest, statCeho)
        if PaR == 1:
            Pp, Rr = evaluateResults(soubAslozkyTest, vysledekTFIDF, soubTest)
            Pdoc2vecLSA.append(Pp)
            Rdoc2vecLSA.append(Rr)

        Accdoc2vecLSA.append(acc)

        # tfidf spojená s doc2vec
        clf = svm.LinearSVC()
        clf.fit(maticeTFIDFD2Vtrain, pozadVystupTrain)
        vysledekTFIDF = clf.predict(maticeTFIDFD2VTest)

        vysShl = []
        for vys in vysledekTFIDF:
            vysShl.append(cisloToShl[vys])
        acc = evaluateResultsAcc(soubAslozkyTest, vysShl, soubTest)
        statCeho = vstupPrac + u'SVMTFIDFdoc2veccrossval' + str(crossValid)
        statistikaVytvorenychShluku(soubAslozkyTest, vysShl, soubTest, statCeho)
        if PaR == 1:
            Pp, Rr = evaluateResults(soubAslozkyTest, vysledekTFIDF, soubTest)
            Ptfidfd2v.append(Pp)
            Rtfidfd2v.append(Rr)
        Acctfidfdoc2vec.append(acc)

    acctfidf = sum(Acctfidf) / float(pocCrossValid)
    acctfidfLSA = sum(AcctfidfLSA) / float(pocCrossValid)
    #accword2vec = sum(Accword2vec) / float(pocCrossValid)
    #accword2vecLSA = sum(Accword2vecLSA) / float(pocCrossValid)
    accdoc2vec = sum(Accdoc2vec) / float(pocCrossValid)
    accdoc2vecLSA = sum(Accdoc2vecLSA) / float(pocCrossValid)
    acctfidfdoc2vec = sum(Acctfidfdoc2vec) / float(pocCrossValid)
    if PaR == 1:
        Pptfidf = sum(PtfidfLSA) / float(pocCrossValid)
        PptfidfLSA = sum(Ptfidf) / float(pocCrossValid)
        Ppdoc2vec = sum(Pdoc2vec) / float(pocCrossValid)
        Ppdoc2vecLSA = sum(Pdoc2vecLSA) / float(pocCrossValid)
        Ptfidfdoc2vec = sum(Ptfidfd2v) / float(pocCrossValid)

        Rrtfidf = sum(Rtfidf) / float(pocCrossValid)
        RrtfidfLSA = sum(RtfidfLSA) / float(pocCrossValid)
        Rrdoc2vec = sum(Rdoc2vec) / float(pocCrossValid)
        Rrdoc2vecLSA = sum(Rdoc2vecLSA) / float(pocCrossValid)
        Rtfidfdoc2vec = sum(Rtfidfd2v) / float(pocCrossValid)
        print 'SVM na TFIDF matici -- Accuracy: ' + str(
            acctfidf)  + '; Precision: ' + str(Pptfidf) + '; Recall: ' + str(Rrtfidf)
        print 'SVM na TFIDF s LSA matici -- Accuracy: ' + str(
            acctfidfLSA) + '; Precision: ' + str(PptfidfLSA) + '; Recall: ' + str(RrtfidfLSA)
        print 'SVM na doc2vec matici -- Accuracy: ' + str(
            accdoc2vec) + '; Precision: ' + str(Ppdoc2vec) + '; Recall: ' + str(Rrdoc2vec)
        print 'SVM na doc2vec s LSA matici -- Accuracy: ' + str(
            accdoc2vecLSA) + '; Precision: ' + str(Ppdoc2vecLSA) + '; Recall: ' + str(Rrdoc2vecLSA)
        print 'SVM s maticí tvořenou spojením TFIDF a doc2vec matic -- Accuracy: ' + str(
            acctfidfdoc2vec) + '; Precision: ' + str(Ptfidfdoc2vec) + '; Recall: ' + str(Rtfidfdoc2vec)
    else:
        print 'SVM na TFIDF matici -- Accuracy: ' + str(acctfidf) # + '; Precision: ' + str(Pptfidf) + '; Recall: ' + str(Rrtfidf)
        print 'SVM na TFIDF s LSA matici -- Accuracy: ' + str(acctfidfLSA) # + '; Precision: ' + str(PptfidfLSA) + '; Recall: ' + str(RrtfidfLSA)
        #print 'SVM na word2vec matici -- Accuracy: ' + str(accword2vec)  # + '; Precision: ' + str(Ppdoc2vec) + '; Recall: ' + str(Rrdoc2vec)
        #print 'SVM na word2vec s LSA matici -- Accuracy: ' + str(accword2vecLSA)  # + '; Precision: ' + str(Ppdoc2vecLSA) + '; Recall: ' + str(Rrdoc2vecLSA)
        print 'SVM na doc2vec matici -- Accuracy: ' + str(accdoc2vec) #+ '; Precision: ' + str(Ppdoc2vec) + '; Recall: ' + str(Rrdoc2vec)
        print 'SVM na doc2vec s LSA matici -- Accuracy: ' + str(accdoc2vecLSA) # + '; Precision: ' + str(Ppdoc2vecLSA) + '; Recall: ' + str(Rrdoc2vecLSA)
        print 'SVM s maticí tvořenou spojením TFIDF a doc2vec matic -- Accuracy: ' + str(acctfidfdoc2vec)  # + '; Precision: ' + str(Ppdoc2vecLSA) + '; Recall: ' + str(Rrdoc2vecLSA)

    KonecnaStatistikaVsechCyklu(vstupPrac + u'SVMTFIDFcrossval')
    KonecnaStatistikaVsechCyklu(vstupPrac + u'SVMTFIDFLSAcrossval')
    KonecnaStatistikaVsechCyklu(vstupPrac + u'SVMdoc2veccrossval')
    KonecnaStatistikaVsechCyklu(vstupPrac + u'SVMdoc2vecLSAcrossval')
    KonecnaStatistikaVsechCyklu(vstupPrac + u'SVMTFIDFdoc2veccrossval')
    file0.write(u'---------- SVM na TFIDF matici ----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Accuracy je: '.encode('utf8') + str(acctfidf).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    if PaR == 1:
        file0.write(u'Precision je: '.encode('utf8') + str(Pptfidf).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Recall je: '.encode('utf8') + str(Rrtfidf).encode('utf8'))
        file0.write('\n'.encode('utf8'))

    file0.write(u'---------- SVM na TFIDF matici se sníženou dimenzí za použití LSA na '.encode('utf8') + str(
        Ncomponents).encode('utf8') + u'----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Accuracy je: '.encode('utf8') + str(acctfidfLSA).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    if PaR == 1:
        file0.write(u'Precision je: '.encode('utf8') + str(PptfidfLSA).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Recall je: '.encode('utf8') + str(RrtfidfLSA).encode('utf8'))
        file0.write('\n'.encode('utf8'))


    '''
    file0.write(u'---------- SVM na word2vec matici ----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Accuracy je: '.encode('utf8') + str(accword2vec).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(
        u'---------- SVM na word2vec matici se sníženou dimenzí za použití LSA na '.encode('utf8') + str(
            Ncomponents).encode(
            'utf8') + u'----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Accuracy je: '.encode('utf8') + str(accword2vecLSA).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    '''
    file0.write(u'---------- SVM na doc2vec matici ----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Accuracy je: '.encode('utf8') + str(accdoc2vec).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    if PaR == 1:
        file0.write(u'Precision je: '.encode('utf8') + str(Ppdoc2vec).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Recall je: '.encode('utf8') + str(Rrdoc2vec).encode('utf8'))
        file0.write('\n'.encode('utf8'))


    file0.write(
        u'---------- SVM na doc2vec matici se sníženou dimenzí za použití LSA na '.encode('utf8') + str(
            Ncomponents).encode(
            'utf8') + u'----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Accuracy je: '.encode('utf8') + str(accdoc2vecLSA).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    if PaR == 1:
        file0.write(u'Precision je: '.encode('utf8') + str(Ppdoc2vecLSA).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Recall je: '.encode('utf8') + str(Rrdoc2vecLSA).encode('utf8'))


    file0.write(u'---------- SVM maticí tvořenou spojením tfidf matice a doc2vec matice s dimenzí '.encode('utf8') + str(2*Ncomponents).encode('utf8') + u'----------'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Accuracy je: '.encode('utf8') + str(acctfidfdoc2vec).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    if PaR == 1:
        file0.write(u'Precision je: '.encode('utf8') + str(Ptfidfdoc2vec).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Recall je: '.encode('utf8') + str(Rtfidfd2v).encode('utf8'))
    file0.close()