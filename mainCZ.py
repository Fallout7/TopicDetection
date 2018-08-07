# -*- coding: utf-8 -*-
# coding: utf-8

from predpriprava import *
from kmeansMethods import UdelejKmeans, UdelejKmeans2matice, UdelejKmeans10Fold
import numpy as np
from evaluation import evaluateResults
from svmClass import *
from LSTMneu import *
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from statistika import *
from upravaDat import *
import sklearn_crfsuite
import pycrfsuite
import scipy


velikostSlovniku = 5000

jazyk = 'czech'
#vstup = 'VstupPrepisy'
#vstup = 'VstupPrepisyVelke'
#vstup = 'VstupPrepisyStereo'
#vstup = 'VstupPrepisyStereoVelke'
vstup = 'VstupPrepisyStereoAmono'
#vstup = 'VstupPrepisyStereoAmonoVelke'

#vstup = 'VstupPrepisyOperator'
#vstup = 'VstupPrepisyOperatorVelke'
#vstup = 'VstupPrepisyStereoOperator'
#vstup = 'VstupPrepisyStereoOperatorVelke'
#vstup = 'VstupPrepisyStereoAmonoOperator'
#vstup = 'VstupPrepisyStereoAmonoOperatorVelke'

#vstup = 'VstupPrepisyZakaznik'
#vstup = 'VstupPrepisyZakaznikVelke'
#vstup = 'VstupPrepisyStereoZakaznik'
#vstup = 'VstupPrepisyStereoZakaznikVelke'
#vstup = 'VstupPrepisyStereoAmonoZakaznik'
#vstup = 'VstupPrepisyStereoAmonoZakaznikVelke'

#vstup = 'VstupShlukyCNO'
#vstup = 'VstupShlukyCNOvelke'
#vstup = 'VstupShlukyCNOstejneVelke'
#vstup = 'VstupShlukyCNOstejnomerne'
# vstup = 'VstupRaw'

#jazyk = 'english'
#vstup = 'Vstup3raw'

SVMANO, PaR = 0, 0
CRFJO = 0
LDAJO = 1
KMEANSJO = 1
NEUJO = 0
train_partition_name = "all" #"train"
test_partition_name = "dev" #"dev"
vypisPr = 0

# nastavení parametrů kmeans a provedení
maxIter = 1000000
tolerancee = 0.00001
nInit = 100
kolikratKmeans = 10
kolikLDA = 50
LDApluscislo = 0
#Ncomponents = int(math.pow(len(textyPracovni), (1.0 / (1 + (np.log(len(textyPracovni)) / 10.0)))))
Ncomponents = 200

#clVyp = '1001424.txt'
#clVyp = '010002'
clVyp = '0303'
if vstup == 'Vstup3rawPorov':
    vstup = 'Vstup3rawPorovTrain'
soubAtextyRaw, soubAslozky = NacteniRawVstupu(vstup)


vycisteneTexty, lemmaTexty, tagsTexty = UpravAVysictiTexty(soubAtextyRaw, vstup, jazyk)


if vypisPr == 1:
    file1 = file(u'ZpracujClanek/priklad.txt', 'w')
    file1.write(codecs.BOM_UTF8)
    print len(soubAtextyRaw)
    print soubAtextyRaw[clVyp]
    file1.write(soubAtextyRaw[clVyp].encode('utf8'))
    file1.write(u'\n'.encode('utf8'))
    file1.write(u'\n'.encode('utf8'))
    print type(vycisteneTexty)


    print u" ".join(vycisteneTexty[clVyp])
    file1.write(u" ".join(vycisteneTexty[clVyp]).encode('utf8'))
    file1.write(u'\n'.encode('utf8'))
    file1.write(u'\n'.encode('utf8'))
    print u" ".join(lemmaTexty[clVyp])
    file1.write(u" ".join(lemmaTexty[clVyp]).encode('utf8'))
    file1.write(u'\n'.encode('utf8'))
    file1.write(u'\n'.encode('utf8'))

# tady se nastaví s čím se bude dále pracovat jestli s cistými texty nebo jejich lemmaty nebo tagy
vstupPrac = vstup +'Lemma'
textyPracovni = lemmaTexty
#vycisteneTexty = []
tagsTexty = []

slovnik, slovnikPole = VytvorVocab(vstupPrac, velikostSlovniku, textyPracovni)
print len(textyPracovni), type(textyPracovni)
print
if vypisPr == 1:
    textCl = lemmaTexty[clVyp]
    textClBezStopSlov = []
    for slovo in textCl:
        if slovnik.has_key(slovo):
            textClBezStopSlov.append(slovo)
    file1.write(u" ".join(textClBezStopSlov).encode('utf8'))
    file1.write(u'\n'.encode('utf8'))
    file1.write(u'\n'.encode('utf8'))
    print u" ".join(textClBezStopSlov)
    print
tfidfMat, nazvySoub = vytvorTFIDF(vstupPrac, textyPracovni, slovnikPole)

# část na výpočet výsledků s LSA
svd = TruncatedSVD(algorithm='randomized', n_components=Ncomponents, n_iter=10, random_state=None,
                   tol=0.0)

normalizer = Normalizer(norm='l2', copy=False)

lsa = make_pipeline(svd, normalizer)
tfidfMatLSA = lsa.fit_transform(tfidfMat)

pozz = 0
if vypisPr == 1:
    for ii in range(len(nazvySoub)):
        if nazvySoub[ii] == clVyp:
            pozz = ii
    vectPrizTFIDF = tfidfMat[pozz].toarray()
    file1.write(u" ".join(slovnikPole[0:10]).encode('utf8'))
    file1.write(u'\n'.encode('utf8'))
    file1.write(u'\n'.encode('utf8'))
    print u" ".join(slovnikPole[0:10])
    print vectPrizTFIDF[0][0:10]
    file1.write(str(vectPrizTFIDF[0][0:10]).encode('utf8'))
    file1.write(u'\n'.encode('utf8'))
    file1.write(u'\n'.encode('utf8'))
    print

#velikost, okno, alphaa = 5000, 10, 0.025
velikost, okno, alphaa, minalphaa, minimalniPocetCetnostiSlov = 500, 0.01, 0.019, 0.019, 5
prvniItr = 1
maxP, maxAlf, maxPref, maxAlfRef = 0.0, 0.0, 0.0, 0.0

for abc in range(1):
    maticeDoc2VecVah = VytvorReprDoc2Vec(vstupPrac, textyPracovni, nazvySoub, velikost, okno, alphaa, minalphaa, minimalniPocetCetnostiSlov)
    maticeWord2VecVah = []
    # část na výpočet výsledků s LSA
    svd = TruncatedSVD(algorithm='randomized', n_components=Ncomponents, n_iter=10, random_state=None,
                       tol=0.0)

    normalizer = Normalizer(norm='l2', copy=False)

    lsa = make_pipeline(svd, normalizer)
    maticeDoc2VecVahLSA = lsa.fit_transform(maticeDoc2VecVah)

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

    if vypisPr == 1:
        print maticeDoc2VecVah[pozz][0:10]
        file1.write(str(maticeDoc2VecVah[pozz][0:10]).encode('utf8'))
        file1.write(u'\n'.encode('utf8'))
        file1.write(u'\n'.encode('utf8'))
        file1.close()
        print

    if SVMANO == 1:
        if vstup == "Vstup3rawPorovTrain":
            UdelejSVMTRAINTEST(tfidfMat, maticeDoc2VecVah, soubAslozky, nazvySoub, Ncomponents, vstupPrac, PaR, slovnik, slovnikPole)
        elif vstup == "Vstup3rawTrain":
            UdelejSVMTRAINTEST2(tfidfMat, maticeDoc2VecVah, soubAslozky, nazvySoub, Ncomponents, vstupPrac, PaR, slovnik,
                               slovnikPole)
        else:
            UdelejSVM(tfidfMat, maticeDoc2VecVah, maticeWord2VecVah, soubAslozky, nazvySoub, Ncomponents, vstupPrac, PaR)

    maticeTFIDFD2V = np.append(tfidfMatLSA, maticeDoc2VecVahLSA, axis=1)

    AccLDA, AccTFIDF, AccTFIDFLSA, Accdoc2vec, Accdoc2vecLSA, AccTFIDFdoc2vec, AccCRFTFIDF, AccCRFTFIDFLSA, AccCRFdoc2vec, AccCRFdoc2vecLSA, AccCRFtfidfdoc2vec = [], [], [], [], [], [], [], [], [], [], []
    AccSVMtfidf, AccSVMtfidfLSA, AccSVMdoc2vec, AccSVMdoc2vecLSA, AccSVMtfidfdoc2vec, AccNeu, AccNeuM = [], [], [], [], [], [], []
    documents = []
    for soubb in nazvySoub:
        documents.append(u' '.join(textyPracovni[soubb]))

    pocShluku = 0
    uzJeSl = {}
    for keyy in soubAslozky:
        if not uzJeSl.has_key(soubAslozky[keyy]):
            uzJeSl[soubAslozky[keyy]] = soubAslozky[keyy]
            pocShluku += 1
    if pocShluku == 1:
        pocShluku = 31

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

    tfidfMat = tfidfMat.toarray()
    print 'Počet shluků je nastaven na: ' + str(pocShluku)
    for crossValid in range(pocCrossValid):
        pocPozTr = crossValid * skokPozZacTrain
        pocPozTs = pocPozTr+pocTrain
        print 'Cyklus: ' + str(crossValid)
        '''
        if pocPozTs <= pocSoub:
            textyPracovniTrain = documents[pocPozTr:pocPozTs]
            soubTrain = nazvySoub[pocPozTr:pocPozTs]
            pozadVystupTrain = pozadVystup[pocPozTr:pocPozTs]
            tfidfTrain = tfidfMat[pocPozTr:pocPozTs]
            tfidfTrainLSA = tfidfMatLSA[pocPozTr:pocPozTs]
            doc2vecTrain = maticeDoc2VecVah[pocPozTr:pocPozTs]
            doc2vecTrainLSA = maticeDoc2VecVahLSA[pocPozTr:pocPozTs]
            tfidfdoc2vecTrainLSA = maticeTFIDFD2V[pocPozTr:pocPozTs]
        else:
            konTr = pocTrain - len(nazvySoub[pocPozTr:pocSoub-1])
            textyPracovniTrain = documents[pocPozTr:pocSoub-1] + documents[0:konTr]
            soubTrain = nazvySoub[pocPozTr:pocSoub-1] + nazvySoub[0:konTr]
            pozadVystupTrain = pozadVystup[pocPozTr:pocSoub-1] + pozadVystup[0:konTr]
            tfidfTrain = np.append(tfidfMat[pocPozTr:pocSoub-1], tfidfMat[0:konTr], axis=0)
            tfidfTrainLSA = np.append(tfidfMatLSA[pocPozTr:pocSoub-1], tfidfMatLSA[0:konTr], axis=0)
            doc2vecTrain = maticeDoc2VecVah[pocPozTr:pocSoub-1] + maticeDoc2VecVah[0:konTr]
            doc2vecTrainLSA = np.append(maticeDoc2VecVahLSA[pocPozTr:pocSoub-1], maticeDoc2VecVahLSA[0:konTr], axis=0)
            tfidfdoc2vecTrainLSA = np.append(maticeTFIDFD2V[pocPozTr:pocSoub - 1], maticeTFIDFD2V[0:konTr], axis=0)

        if pocPozTs + pocTest <= pocSoub:
            textyPracovniTest = documents[pocPozTs:pocPozTs + pocTest]
            soubTest = nazvySoub[pocPozTs:pocPozTs + pocTest]
            tfidfTest = tfidfMat[pocPozTs:pocPozTs + pocTest]
            tfidfTestLSA = tfidfMatLSA[pocPozTs:pocPozTs + pocTest]
            doc2vecTest = maticeDoc2VecVah[pocPozTs:pocPozTs + pocTest]
            doc2vecTestLSA = maticeDoc2VecVahLSA[pocPozTs:pocPozTs + pocTest]
            tfidfdoc2vecTestLSA = maticeTFIDFD2V[pocPozTs:pocPozTs + pocTest]
            pozadVystupTest = pozadVystup[pocPozTs:pocPozTs + pocTest]
        else:
            konTs = pocTest - len(nazvySoub[pocPozTs:pocSoub - 1])
            textyPracovniTest = documents[pocPozTs:pocSoub - 1] + documents[0:konTs]
            soubTest = nazvySoub[pocPozTs:pocSoub - 1] + nazvySoub[0:konTs]
            tfidfTest = np.append(tfidfMat[pocPozTs:pocSoub - 1], tfidfMat[0:konTs], axis=0)
            tfidfTestLSA = np.append(tfidfMatLSA[pocPozTs:pocSoub - 1], tfidfMatLSA[0:konTs], axis=0)
            doc2vecTest = maticeDoc2VecVah[pocPozTs:pocSoub - 1] + maticeDoc2VecVah[0:konTs]
            doc2vecTestLSA = np.append(maticeDoc2VecVahLSA[pocPozTs:pocSoub - 1], maticeDoc2VecVahLSA[0:konTs], axis=0)
            tfidfdoc2vecTestLSA = np.append(maticeTFIDFD2V[pocPozTs:pocSoub - 1], maticeTFIDFD2V[0:konTs], axis=0)
            pozadVystupTest = pozadVystup[pocPozTs:pocSoub - 1] + pozadVystup[0:konTs]

        soubAslozkyTest = {}
        for soub in soubTest:
            soubAslozkyTest[soub] = soubAslozky[soub]
        '''

        # TF-IDF matice svm a neu
        tfidfTrain, tfidfTest, pozadVystupTrainTFIDF, soubAslozkyTestTFIDF, pozadVystupTestTFIDF, pocShluku, soubTestTFIDF, textyPracovniTrain, textyPracovniTest = ExtrakceCastiMaticeUnsup(
            nazvySoub, soubAslozky, pocTrainShl, tfidfMat, pozadVystup, skokPozZacTrainShl, crossValid, propojeni, documents)
        acc = svmCast(tfidfTrain, pozadVystupTrainTFIDF, tfidfTest, soubAslozkyTestTFIDF, soubTestTFIDF)
        AccSVMtfidf.append(acc)
        print 'Acc SVM s tfidf maticí pro cyklus ' + str(crossValid) + ' vychází: ' + str(acc)
        # TF-IDF matice snížená LSA svm a neu
        tfidfTrainLSA, tfidfTestLSA, pozadVystupTrain, soubAslozkyTestTFIDFLSA, pozadVystupTestTFIDFLSA, pocShluku, soubTestTFIDFLSA, textyPracovniTrain, textyPracovniTest = ExtrakceCastiMaticeUnsup(
            nazvySoub, soubAslozky, pocTrainShl, tfidfMatLSA, pozadVystup, skokPozZacTrainShl, crossValid,
            propojeni, documents)
        acc = svmCast(tfidfTrainLSA, pozadVystupTrain, tfidfTestLSA, soubAslozkyTestTFIDFLSA, soubTestTFIDFLSA)
        AccSVMtfidfLSA.append(acc)
        print 'Acc SVM s tfidfLSA maticí pro cyklus ' + str(crossValid) + ' vychází: ' + str(acc)
        # doc2vec matice svm a neu
        doc2vecTrain, doc2vecTest, pozadVystupTrain, soubAslozkyTestdoc2vec, pozadVystupTestdoc2vec, pocShluku, soubTestdoc2vec, textyPracovniTrain, textyPracovniTest = ExtrakceCastiMaticeUnsup(
            nazvySoub, soubAslozky, pocTrainShl, maticeDoc2VecVah, pozadVystup, skokPozZacTrainShl, crossValid,
            propojeni, documents)
        acc = svmCast(doc2vecTrain, pozadVystupTrain, doc2vecTest, soubAslozkyTestdoc2vec, soubTestdoc2vec)
        AccSVMdoc2vec.append(acc)
        print 'Acc SVM s doc2vec maticí pro cyklus ' + str(crossValid) + ' vychází: ' + str(acc)
        # doc2vec matice snížená LSA svm a neu
        doc2vecTrainLSA, doc2vecTestLSA, pozadVystupTrain, soubAslozkyTestdoc2vecLSA, pozadVystupTestdoc2vecLSA, pocShluku, soubTestdoc2vecLSA, textyPracovniTrain, textyPracovniTest = ExtrakceCastiMaticeUnsup(
            nazvySoub, soubAslozky, pocTrainShl, maticeDoc2VecVahLSA, pozadVystup, skokPozZacTrainShl, crossValid,
            propojeni, documents)
        acc = svmCast(doc2vecTrainLSA, pozadVystupTrain, doc2vecTestLSA, soubAslozkyTestdoc2vecLSA, soubTestdoc2vecLSA)
        Accdoc2vecLSA.append(acc)
        print 'Acc SVM s doc2vecLSA maticí pro cyklus ' + str(crossValid) + ' vychází: ' + str(acc)
        # tfidf a doc2vec matice svm a neu
        tfidfdoc2vecTrainLSA, tfidfdoc2vecTestLSA, pozadVystupTrain, soubAslozkyTestTFIDFdoc2vec, pozadVystupTestTFIDFdoc2vec, pocShluku, soubTestTFIDFdoc2vec, textyPracovniTrain, textyPracovniTest = ExtrakceCastiMaticeUnsup(
            nazvySoub, soubAslozky, pocTrainShl, maticeTFIDFD2V, pozadVystup, skokPozZacTrainShl, crossValid,
            propojeni, documents)
        acc = svmCast(tfidfdoc2vecTrainLSA, pozadVystupTrain, tfidfdoc2vecTestLSA, soubAslozkyTestTFIDFdoc2vec, soubTestTFIDFdoc2vec)
        AccTFIDFdoc2vec.append(acc)
        print 'Acc SVM s tfidfdoc2vec maticí pro cyklus ' + str(crossValid) + ' vychází: ' + str(acc)

        if CRFJO == 1:
            ytr = []
            for sf in pozadVystupTrainTFIDF:
                ytr.append(str(sf)*5000)

            crf = sklearn_crfsuite.CRF(
                algorithm='lbfgs',
                c1=0.1,
                c2=0.1,
                max_iterations=1000,
                all_possible_transitions=True
            )
            crf.fit(tfidfTrain, ytr)
            y_predPom = crf.predict(tfidfTest)
            y_pred = []
            for iii in range(len(y_predPom)):
                y_pred.append(y_predPom[iii][0])
            vys = evaluateResultsAcc2(soubAslozkyTestTFIDF, y_pred, soubTestTFIDF)
            AccCRFTFIDF.append(vys)
            print 'CRF TFIDF: ' + str(evaluateResultsAcc2(soubAslozkyTestTFIDF, y_pred, soubTestTFIDF))

            ytr = []
            for sf in pozadVystupTrainTFIDF:
                ytr.append(str(sf) * 200)
            crf = sklearn_crfsuite.CRF(
                algorithm='lbfgs',
                c1=0.1,
                c2=0.1,
                max_iterations=1000,
                all_possible_transitions=True
            )
            crf.fit(tfidfTrainLSA, ytr)
            y_predPom = crf.predict(tfidfTestLSA)
            y_pred = []
            for iii in range(len(y_predPom)):
                y_pred.append(y_predPom[iii][0])
            vys = evaluateResultsAcc2(soubAslozkyTestTFIDFLSA, y_pred, soubTestTFIDFLSA)
            AccCRFTFIDFLSA.append(vys)
            print 'CRF TFIDF LSA: ' + str(vys)

            ytr = []
            for sf in pozadVystupTrainTFIDF:
                ytr.append(str(sf) * 500)
            crf = sklearn_crfsuite.CRF(
                algorithm='lbfgs',
                c1=0.1,
                c2=0.1,
                max_iterations=1000,
                all_possible_transitions=True
            )
            crf.fit(doc2vecTrain, ytr)
            y_predPom = crf.predict(doc2vecTest)
            y_pred = []
            for iii in range(len(y_predPom)):
                y_pred.append(y_predPom[iii][0])
            vys = evaluateResultsAcc2(soubAslozkyTestdoc2vec, y_pred, soubTestdoc2vec)
            AccCRFTFIDF.append(vys)
            print 'CRF doc2vec: ' + str(vys)

            ytr = []
            for sf in pozadVystupTrainTFIDF:
                ytr.append(str(sf) * 200)
            crf = sklearn_crfsuite.CRF(
                algorithm='lbfgs',
                c1=0.1,
                c2=0.1,
                max_iterations=1000,
                all_possible_transitions=True
            )
            crf.fit(doc2vecTrainLSA, ytr)
            y_predPom = crf.predict(doc2vecTestLSA)
            y_pred = []
            for iii in range(len(y_predPom)):
                y_pred.append(y_predPom[iii][0])
            vys = evaluateResultsAcc2(soubAslozkyTestdoc2vecLSA, y_pred, soubTestdoc2vecLSA)
            AccCRFTFIDF.append(vys)
            print 'CRF doc2vec LSA: ' + str(vys)

            ytr = []
            for sf in pozadVystupTrainTFIDF:
                ytr.append(str(sf) * 400)
            crf = sklearn_crfsuite.CRF(
                algorithm='lbfgs',
                c1=0.1,
                c2=0.1,
                max_iterations=1000,
                all_possible_transitions=True
            )
            crf.fit(tfidfdoc2vecTrainLSA, ytr)
            y_predPom = crf.predict(tfidfdoc2vecTestLSA)
            y_pred = []
            for iii in range(len(y_predPom)):
                y_pred.append(y_predPom[iii][0])
            vys = evaluateResultsAcc2(soubAslozkyTestTFIDFdoc2vec, y_pred, soubTestTFIDFdoc2vec)
            AccCRFTFIDF.append(vys)
            print 'CRF tfidf doc2vec: ' + str(vys)

        if LDAJO == 1:

            tf_vectorizer = CountVectorizer(max_df=1.0, min_df=1, max_features=2000)
            tfTrain = tf_vectorizer.fit_transform(textyPracovniTrain)

            tfTest = tf_vectorizer.transform(textyPracovniTest)

            no_topics = pocShluku
            maxHod = 0.0
            hodOffset = 0.0
            for ijj in range(kolikLDA):
                i = ijj + LDApluscislo
                lda_model = LatentDirichletAllocation(n_topics=no_topics, max_iter=10, learning_method='online',
                                                      learning_offset=float(i), random_state=1)
                lda_model.fit(tfTrain)
                lda_W = lda_model.transform(tfTest)
                vysledkyLDA = []
                for iii in range(len(lda_W)):
                    shlukyy = lda_W[iii]
                    maxx = 0.0
                    shluk = 0
                    for iij in range(len(shlukyy)):
                        shl = shlukyy[iij]
                        if maxx < shl:
                            maxx = shl
                            shluk = iij
                    vysledkyLDA.append(shluk)
                acc = evaluateResultsAcc2(soubAslozkyTestTFIDF, vysledkyLDA, soubTestTFIDF)

                if maxHod < acc:
                    maxHod = acc
                    hodOffset = i
            print 'Otestované hodnoty pro LDA, learnint_offset nastaven na: ' + str(hodOffset)

            lda_model = LatentDirichletAllocation(n_topics=no_topics, max_iter=20, learning_method='online',
                                                  learning_offset=hodOffset, random_state=1)
            lda_model.fit(tfTrain)
            lda_W = lda_model.transform(tfTest)
            vysledkyLDA = []
            for iii in range(len(lda_W)):
                shlukyy = lda_W[iii]
                maxx = 0.0
                shluk = 0
                for iij in range(len(shlukyy)):
                    shl = shlukyy[iij]
                    if maxx < shl:
                        maxx = shl
                        shluk = iij
                vysledkyLDA.append(shluk)
            acc = evaluateResultsAcc2(soubAslozkyTestTFIDF, vysledkyLDA, soubTestTFIDF)
            print 'Acc LDA pro cyklus ' + str(crossValid) + ' vychází: ' + str(acc)
            statCeho = vstupPrac + u'LDA' + str(crossValid)
            statistikaVytvorenychShluku(soubAslozkyTestTFIDF, vysledkyLDA, soubTestTFIDF, statCeho)

            AccLDA.append(acc)

        if KMEANSJO == 1:
            print 'Trénování a testování kmeans s TFIDF maticí a sníženou dimenzí na ' + str(Ncomponents)
            accPomTFIDF = []
            accPomTFIDFLSA = []
            for ii in range(kolikratKmeans):
                vysledekClusteringu, vysledekClusteringuLSA, clustering, clusteringLSA = UdelejKmeans10Fold(vstupPrac+'TFIDF', tfidfTrain, tfidfTest, tfidfTrainLSA, tfidfTestLSA, maxIter, tolerancee, nInit, ii, pocShluku, crossValid)

                '''
                vystup = 'VypisyVyslednychShluku/' + vstupPrac + '/'
                try:
                    os.stat(vystup)
                except:
                    os.mkdir(vystup)
                for ij in range(len(vysledekClusteringu)):
                    soub = nazvySoub[ij]
                    shluk = vysledekClusteringu[ij]
                    try:
                        os.stat(vystup + str(shluk) + '/')
                    except:
                        os.mkdir(vystup + str(shluk) + '/')
                    shutil.copyfile(vstup + '/' + soubAslozky[soub] + '/' + soub, vystup + str(shluk) + '/' + soub)
                '''
                accPomTFIDF.append(evaluateResultsAcc2(soubAslozkyTestTFIDF, vysledekClusteringu, soubTestTFIDF))
                accPomTFIDFLSA.append(evaluateResultsAcc2(soubAslozkyTestTFIDFLSA, vysledekClusteringuLSA, soubTestTFIDFLSA))

                statCeho = vstupPrac + u'K-meansTFIDF' + str(crossValid) + str(ii)
                statistikaVytvorenychShluku(soubAslozkyTestTFIDF, vysledekClusteringu, soubTestTFIDF, statCeho)
                statCeho = vstupPrac + u'K-meansLSATFIDF' + str(crossValid) + str(ii)
                statistikaVytvorenychShluku(soubAslozkyTestTFIDFLSA, vysledekClusteringuLSA, soubTestTFIDFLSA, statCeho)
            print 'Acc K-means s TFIDF mat pro cyklus ' + str(crossValid) + ' vychází: ' + str(sum(accPomTFIDF)/10.0)
            print 'Acc K-means s TFIDFLSA mat pro cyklus ' + str(crossValid) + ' vychází: ' + str(sum(accPomTFIDFLSA) / 10.0)
            AccTFIDF.append(sum(accPomTFIDF)/10.0)
            AccTFIDFLSA.append(sum(accPomTFIDFLSA)/10.0)

            # nastavení parametrů kmeans a provedení


            print 'Trénování a testování kmeans s doc2vec maticí a nastavenou hodnotou alpha: ' + str(alphaa)
            accPomdoc2vec = []
            accPomdoc2vecLSA = []
            for ii in range(kolikratKmeans):
                vysledekClusteringu, vysledekClusteringuLSA, clustering, clusteringLSA = UdelejKmeans10Fold(
                    vstupPrac + 'doc2vec', doc2vecTrain, doc2vecTest, doc2vecTrainLSA, doc2vecTestLSA, maxIter, tolerancee, nInit,
                    ii, pocShluku, crossValid)

                accPomdoc2vec.append(evaluateResultsAcc2(soubAslozkyTestdoc2vec, vysledekClusteringu, soubTestdoc2vec))
                accPomdoc2vecLSA.append(evaluateResultsAcc2(soubAslozkyTestdoc2vecLSA, vysledekClusteringuLSA, soubTestdoc2vecLSA))

                statCeho = vstupPrac + u'K-meansdoc2vec' + str(crossValid) + str(ii)
                statistikaVytvorenychShluku(soubAslozkyTestdoc2vec, vysledekClusteringu, soubTestdoc2vec, statCeho)
                statCeho = vstupPrac + u'K-meansLSAdoc2vec' + str(crossValid) + str(ii)
                statistikaVytvorenychShluku(soubAslozkyTestdoc2vecLSA, vysledekClusteringuLSA, soubTestdoc2vecLSA, statCeho)

            Accdoc2vec.append(sum(accPomdoc2vec) / 10.0)
            Accdoc2vecLSA.append(sum(accPomdoc2vecLSA) / 10.0)
            print 'Acc K-means s doc2vec mat pro cyklus ' + str(crossValid) + ' vychází: ' + str(sum(accPomdoc2vec) / 10.0)
            print 'Acc K-means s doc2vecLSA mat pro cyklus ' + str(crossValid) + ' vychází: ' + str(sum(accPomdoc2vecLSA) / 10.0)

            print 'Trénování a testování kmeans s maticí vytvořenou spojením TFIDF a doc2vec matic'
            acctfidfdoc2vecPom = []
            for ii in range(kolikratKmeans):
                vysledekClusteringu, vysledekClusteringuLSA, clustering, clusteringLSA = UdelejKmeans10Fold(
                    vstupPrac + 'TFIDFdoc2vec', tfidfdoc2vecTrainLSA, tfidfdoc2vecTestLSA, tfidfdoc2vecTrainLSA, tfidfdoc2vecTestLSA, maxIter,
                    tolerancee, nInit,
                    ii, pocShluku, crossValid)

                acctfidfdoc2vecPom.append(evaluateResultsAcc2(soubAslozkyTestTFIDFdoc2vec, vysledekClusteringu, soubTestTFIDFdoc2vec))
                statCeho = vstupPrac + u'K-meansdocTFIDF2vec' + str(ii)
                statistikaVytvorenychShluku(soubAslozkyTestTFIDFdoc2vec, vysledekClusteringu, soubTestTFIDFdoc2vec, statCeho)
            AccTFIDFdoc2vec.append(sum(acctfidfdoc2vecPom) / 10.0)
            print 'Acc K-means s tfidf + doc2vec mat pro cyklus ' + str(crossValid) + ' vychází: ' + str(
                sum(acctfidfdoc2vecPom) / 10.0)
        if NEUJO == 1:
            epochs = 200
            batch_size = 100
            dropout = 0.9
            dropout2 = 0.9
            un = 500
            un2 = 300
            un3 = 640
            un4 = 320
            vysledekClusteringu, vysledekClusteringuLSA, clustering, clusteringLSA = UdelejKmeans10Fold(
                vstupPrac + 'TFIDF', tfidfTrain, tfidfTest, tfidfTrainLSA, tfidfTestLSA, maxIter, tolerancee, nInit, 0,
                pocShluku, crossValid)
            accK, accNeu = neuProved(tfidfTrain, tfidfTest, vysledekClusteringu, pozadVystupTestTFIDF, pocShluku,
                                     train_partition_name, epochs, batch_size, soubAslozkyTestTFIDF, soubTestTFIDF, dropout,
                                     dropout2, un,
                                     un2)
            AccNeu.append(accK)
            AccNeuM.append(accNeu)

    KonecnaStatistikaVsechCyklu(vstupPrac + u'LDA')
    KonecnaStatistikaVsechCyklu(vstupPrac + u'K-meansTFIDF')
    KonecnaStatistikaVsechCyklu(vstupPrac + u'K-meansLSATFIDF')
    KonecnaStatistikaVsechCyklu(vstupPrac + u'K-meansdoc2vec')
    KonecnaStatistikaVsechCyklu(vstupPrac + u'K-meansLSAdoc2vec')
    KonecnaStatistikaVsechCyklu(vstupPrac + u'K-meansdocTFIDF2vec')
    file0 = file(u'Výsledky' + vstupPrac, 'w')
    file0.write(codecs.BOM_UTF8)
    file0.write(
        u'Výsledky SVM s TF-IDF maticí a 10-fold cross validací na datech '.encode('utf8') + vstupPrac.encode(
            'utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Acc je: '.encode('utf8') + str(sum(AccSVMtfidf) / float((pocCrossValid))).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(
        u'Výsledky SVM s TF-IDF maticí a 10-fold cross validací a sníženou dimenzí '.encode('utf8') + str(
            Ncomponents).encode(
            'utf8') + u' na na datech '.encode('utf8') + vstupPrac.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Acc je: '.encode('utf8') + str(sum(AccSVMtfidfLSA) / float((pocCrossValid))).encode('utf8'))
    file0.write('\n'.encode('utf8'))
    file0.write('\n'.encode('utf8'))

    file0.write(u'Výsledky SVM s doc2vec maticí o velikosti'.encode('utf8') + str(velikost).encode(
        'utf8') + u' a 10-fold cross validací na datech '.encode('utf8') + vstupPrac.encode(
        'utf8') + u' s hodnotami alpha a minalpha: '.encode('utf8') + str(alphaa).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Acc je: '.encode('utf8') + str(sum(AccSVMdoc2vec) / float((pocCrossValid))).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Výsledky SVM s doc2vec maticí a sníženou dimenzí '.encode('utf8') + str(Ncomponents).encode(
        'utf8') + u' a 10-fold cross validací na na datech '.encode('utf8') + vstupPrac.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Acc je: '.encode('utf8') + str(sum(AccSVMdoc2vecLSA) / float((pocCrossValid))).encode('utf8'))
    file0.write('\n'.encode('utf8'))
    file0.write('\n'.encode('utf8'))
    file0.write(
        u'Výsledky SVM s maticí vytvořenou spojením matic tfidf a doc2vec a 10-fold cross validací s počtem příznaků '.encode(
            'utf8') + str(2 * Ncomponents).encode(
            'utf8') + u' na datech '.encode('utf8') + vstupPrac.encode(
            'utf8') + u' s hodnotami alpha a minalpha: '.encode(
            'utf8') + str(alphaa).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Acc je: '.encode('utf8') + str(sum(AccSVMtfidfdoc2vec) / float((pocCrossValid))).encode('utf8'))
    file0.write('\n'.encode('utf8'))
    file0.write('\n'.encode('utf8'))



    file0.write(
        u'Výsledky LDA 10-fold cross validací na datech '.encode('utf8') + vstupPrac.encode(
            'utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Acc je: '.encode('utf8') + str(sum(AccLDA) / float((pocCrossValid))).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Výsledky K-means s TF-IDF maticí a 10-fold cross validací na datech '.encode('utf8') + vstupPrac.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Acc je: '.encode('utf8') + str(sum(AccTFIDF) / float((pocCrossValid))).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(
        u'Výsledky K-means s TF-IDF maticí a 10-fold cross validací a sníženou dimenzí '.encode('utf8') + str(Ncomponents).encode(
            'utf8') + u' na na datech '.encode('utf8') + vstupPrac.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Acc je: '.encode('utf8') + str(sum(AccTFIDFLSA) / float((pocCrossValid))).encode('utf8'))
    file0.write('\n'.encode('utf8'))
    file0.write('\n'.encode('utf8'))

    file0.write(u'Výsledky K-means s doc2vec maticí o velikosti'.encode('utf8') + str(velikost).encode(
        'utf8') + u' a 10-fold cross validací na datech '.encode('utf8') + vstupPrac.encode(
        'utf8') + u' s hodnotami alpha a minalpha: '.encode('utf8') + str(alphaa).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Acc je: '.encode('utf8') + str(sum(Accdoc2vec) / float((pocCrossValid))).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Výsledky K-means s doc2vec maticí a sníženou dimenzí '.encode('utf8') + str(Ncomponents).encode(
        'utf8') + u' a 10-fold cross validací na na datech '.encode('utf8') + vstupPrac.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Acc je: '.encode('utf8') + str(sum(Accdoc2vecLSA) / float((pocCrossValid))).encode('utf8'))
    file0.write('\n'.encode('utf8'))
    file0.write('\n'.encode('utf8'))

    file0.write(u'Výsledky K-means s maticí vytvořenou spojením matic tfidf a doc2vec a 10-fold cross validací s počtem příznaků '.encode(
        'utf8') + str(2 * Ncomponents).encode(
        'utf8') + u' na datech '.encode('utf8') + vstupPrac.encode(
        'utf8') + u' s hodnotami alpha a minalpha: '.encode(
        'utf8') + str(alphaa).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Acc je: '.encode('utf8') + str(sum(AccTFIDFdoc2vec) / float((pocCrossValid))).encode('utf8'))
    file0.write('\n'.encode('utf8'))
    file0.write('\n'.encode('utf8'))

    if CRFJO == 1:
        file0.write(
            u'Výsledky CRF s TF-IDF maticí a 10-fold cross validací na datech '.encode('utf8') + vstupPrac.encode(
                'utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Acc je: '.encode('utf8') + str(sum(AccCRFTFIDF) / float((pocCrossValid))).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(
            u'Výsledky CRF s TF-IDF maticí a 10-fold cross validací a sníženou dimenzí '.encode('utf8') + str(
                Ncomponents).encode(
                'utf8') + u' na na datech '.encode('utf8') + vstupPrac.encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Acc je: '.encode('utf8') + str(sum(AccCRFTFIDFLSA) / float((pocCrossValid))).encode('utf8'))
        file0.write('\n'.encode('utf8'))
        file0.write('\n'.encode('utf8'))

        file0.write(u'Výsledky CRF s doc2vec maticí o velikosti'.encode('utf8') + str(velikost).encode(
            'utf8') + u' a 10-fold cross validací na datech '.encode('utf8') + vstupPrac.encode(
            'utf8') + u' s hodnotami alpha a minalpha: '.encode('utf8') + str(alphaa).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Acc je: '.encode('utf8') + str(sum(AccCRFdoc2vec) / float((pocCrossValid))).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Výsledky CRF s doc2vec maticí a sníženou dimenzí '.encode('utf8') + str(Ncomponents).encode(
            'utf8') + u' a 10-fold cross validací na na datech '.encode('utf8') + vstupPrac.encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Acc je: '.encode('utf8') + str(sum(AccCRFdoc2vecLSA) / float((pocCrossValid))).encode('utf8'))
        file0.write('\n'.encode('utf8'))
        file0.write('\n'.encode('utf8'))

        file0.write(
            u'Výsledky CRF s maticí vytvořenou spojením matic tfidf a doc2vec a 10-fold cross validací s počtem příznaků '.encode(
                'utf8') + str(2 * Ncomponents).encode(
                'utf8') + u' na datech '.encode('utf8') + vstupPrac.encode(
                'utf8') + u' s hodnotami alpha a minalpha: '.encode(
                'utf8') + str(alphaa).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Acc je: '.encode('utf8') + str(sum(AccCRFtfidfdoc2vec) / float((pocCrossValid))).encode('utf8'))
        file0.write('\n'.encode('utf8'))
        file0.write('\n'.encode('utf8'))

    if NEUJO == 1:
        file0.write(
            u'Výsledky Neu s TF-IDF maticí a 10-fold cross validací na datech '.encode('utf8') + vstupPrac.encode(
                'utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Acc je: '.encode('utf8') + str(sum(AccNeu) / float((pocCrossValid))).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(
            u'Výsledky NeuM s TF-IDF maticí a 10-fold cross validací a sníženou dimenzí '.encode('utf8') + str(
                Ncomponents).encode(
                'utf8') + u' na na datech '.encode('utf8') + vstupPrac.encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Acc je: '.encode('utf8') + str(sum(AccNeuM) / float((pocCrossValid))).encode('utf8'))
        file0.write('\n'.encode('utf8'))
        file0.write('\n'.encode('utf8'))

    file0.close()