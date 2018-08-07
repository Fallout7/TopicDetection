# -*- coding: utf-8 -*-
# coding: utf-8

from predpriprava import *
from kmeansMethods import UdelejKmeans, UdelejKmeans2matice, UdelejKmeansTS, UdelejKmeans2maticeTS
import numpy as np
from evaluation import evaluateResults
from svmClass import *
from LSTMneu import *

import scipy

velikostSlovniku = 5000
jazyk = 'english'
#vstupTrain = 'Vstup3rawPorovTrain'
#vstupTest = ["Vstup3rawPorovTest50"]
#vstupTest = ["Vstup3rawPorovTest200"]
#vstupTest = ["Vstup3rawPorovTest350"]
vstupTrain = "Vstup3rawTrain"
vstupTest = ["Vstup3rawTest"]

SVMANO, PaR = 1, 0
LSTMNEU = 0
predTrain = 0
KMEANSJO = 1

soubAtextyRaw, soubAslozky = NacteniRawVstupu(vstupTrain)
print len(soubAtextyRaw)
vycisteneTexty, lemmaTexty, tagsTexty = UpravAVysictiTexty(soubAtextyRaw, vstupTrain, jazyk)

vstupPrac = vstupTrain + 'CistyText'
textyPracovni = vycisteneTexty
lemmaTexty = []
tagsTexty = []
textyTrain = textyPracovni
prvniItr = 1
for testSet in vstupTest:
    if testSet == "Vstup3rawPorovTest50":
        Ncomponents = 50
    else:
        Ncomponents = 200
    slovnik, slovnikPole = VytvorVocab(vstupPrac, velikostSlovniku, textyPracovni)

    tfidfMat, nazvySoub = vytvorTFIDF(vstupPrac, textyPracovni, slovnikPole)
    nazvySoubTrain = nazvySoub
    # velikost, okno, alphaa = 5000, 10, 0.025
    velikost, okno, alphaa, minalphaa, minimalniPocetCetnostiSlov = 5000, 0.01, 0.019, 0.019, 5
    prvniItr = 1
    maxP, maxAlf, maxPref, maxAlfRef = 0.0, 0.0, 0.0, 0.0

    # nastavení parametrů kmeans a provedení
    maticePouzVah = tfidfMat
    maxIter = 1000000
    tolerancee = 0.00001
    nInit = 100
    #Ncomponents = int(math.pow(len(textyPracovni), (1.0 / (1 + (np.log(len(textyPracovni)) / 10.0)))))

    if prvniItr == 1:
        prvniItr = 0
        print '---------------------skocilo to sem ----------------------------------'
        if SVMANO == 1:
            if vstupTrain == 'Vstup3rawPorovTrain':
                UdelejSVMTRAINTEST(tfidfMat, textyPracovni, soubAslozky, nazvySoub, Ncomponents, vstupPrac, PaR, slovnik, slovnikPole)
                print 'provedeno'
            else:
                UdelejSVMTRAINTEST2(tfidfMat, textyPracovni, soubAslozky, nazvySoub, Ncomponents, vstupPrac, PaR, slovnik, slovnikPole)
                print 'dasggaadfsgfg'

    print '--------------- Provádí se K-means ----------------------------'
    if KMEANSJO == 1:

        file0 = file(u'Výsledky' + vstupPrac + testSet, 'w')
        file0.write(codecs.BOM_UTF8)

        soubAtextyRaw, soubAslozky = NacteniRawVstupu(testSet)
        vycisteneTexty, lemmaTexty, tagsTexty = UpravAVysictiTexty(soubAtextyRaw, testSet, 'english')
        vstupPrac = testSet + 'CistyText'
        textyPracovni = vycisteneTexty

        tfidfMatTest, nazvySoub = vytvorTFIDF(vstupPrac, textyPracovni, slovnikPole)

        velikost, okno, alphaa, minalphaa, minimalniPocetCetnostiSlov = 5000, 0.01, 0.019, 0.019, 5

        maticeDoc2VecVah, maticeDoc2VecVahTest = VytvorReprDoc2VecTrainTest(vstupPrac+'KMEANS', textyTrain, textyPracovni, nazvySoubTrain, nazvySoub, velikost, okno, alphaa,
                                                     minalphaa, minimalniPocetCetnostiSlov)

        print 'Trénování a testování kmeans s TFIDF maticí.'
        P, R, Plsa, Rlsa = [], [], [], []
        for ii in range(2):
            vysledekClusteringu, vysledekClusteringuLSA, clustering, clusteringLSA = UdelejKmeansTS(testSet + 'TFIDF',
                                                                                                  soubAslozky, maxIter,
                                                                                                  tolerancee, nInit,
                                                                                                  Ncomponents,
                                                                                                  tfidfMat, tfidfMatTest, ii)

            Pp, Rr = evaluateResults(soubAslozky, vysledekClusteringu, nazvySoub)
            Pplsa, Rrlsa = evaluateResults(soubAslozky, vysledekClusteringuLSA, nazvySoub)
            P.append(Pp)
            R.append(Rr)
            Plsa.append(Pplsa)
            Rlsa.append(Rrlsa)
            # print Pp, Rr, Pplsa, Rrlsa

        print sum(P) / float(len(P)), sum(R) / float(len(R))
        print sum(Plsa) / float(len(Plsa)), sum(Rlsa) / float(len(Rlsa))

        # nastavení parametrů kmeans a provedení
        file0.write(u'Výsledky K-means s TF-IDF maticí'.encode('utf8') + u' na na datech '.encode('utf8') + vstupPrac.encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Precision je: '.encode('utf8') + str(sum(P) / float(len(R))).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Recall je: '.encode('utf8') + str(sum(Rlsa) / float(len(Rlsa))).encode('utf8'))
        file0.write('\n'.encode('utf8'))
        file0.write('\n'.encode('utf8'))

        file0.write(u'Výsledky K-means s TF-IDF maticí a sníženou dimenzí '.encode('utf8') + str(Ncomponents).encode(
            'utf8') + u' na na datech '.encode('utf8') + vstupPrac.encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Precision je: '.encode('utf8') + str(sum(Plsa) / float(len(Plsa))).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Recall je: '.encode('utf8') + str(sum(Rlsa) / float(len(Rlsa))).encode('utf8'))
        file0.write('\n'.encode('utf8'))
        file0.write('\n'.encode('utf8'))



        print 'Trénování a testování kmeans s doc2vec maticí a nastavenou hodnotou alpha: ' + str(alphaa)
        maticePouzVah = maticeDoc2VecVah
        #Ncomponents = int(math.pow(len(textyPracovni), (1.0 / (1 + (np.log(len(textyPracovni)) / 10.0)))))
        P, R, Plsa, Rlsa = [], [], [], []
        for ii in range(2):
            vysledekClusteringu, vysledekClusteringuLSA, clustering, clusteringLSA = UdelejKmeansTS(testSet+'doc2vec'+str(alphaa), soubAslozky, maxIter, tolerancee, nInit, Ncomponents, maticeDoc2VecVah, maticeDoc2VecVahTest, ii)

            Pp, Rr = evaluateResults(soubAslozky, vysledekClusteringu, nazvySoub)
            Pplsa, Rrlsa = evaluateResults(soubAslozky, vysledekClusteringuLSA, nazvySoub)
            P.append(Pp)
            R.append(Rr)
            Plsa.append(Pplsa)
            Rlsa.append(Rrlsa)
            #print Pp, Rr, Pplsa, Rrlsa

        Ppom = sum(P) / float(len(P))
        PpomLSa = sum(Plsa) / float(len(Plsa))
        if Ppom > maxP:
            maxP = Ppom
            maxAlf = alphaa
        if PpomLSa > maxP:
            maxP = PpomLSa
            maxAlf = alphaa
        print sum(P) / float(len(P)), sum(R) / float(len(R))
        print sum(Plsa) / float(len(Plsa)), sum(Rlsa) / float(len(Rlsa))
        file0.write(u'Výsledky K-means s doc2vec maticí o velikosti'.encode('utf8') + str(velikost).encode('utf8') + u' na datech '.encode('utf8') + vstupPrac.encode('utf8') + u' s hodnotami alpha a minalpha: '.encode('utf8') + str(alphaa).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Precision je: '.encode('utf8') + str(sum(P) / float(len(P))).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Recall je: '.encode('utf8') + str(sum(R) / float(len(R))).encode('utf8'))
        file0.write('\n'.encode('utf8'))
        file0.write(u'Výsledky K-means s doc2vec maticí a sníženou dimenzí '.encode('utf8') + str(Ncomponents).encode('utf8') + u' na na datech '.encode('utf8') + vstupPrac.encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Precision je: '.encode('utf8') + str(sum(Plsa) / float(len(Plsa))).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Recall je: '.encode('utf8') + str(sum(Rlsa) / float(len(Rlsa))).encode('utf8'))
        file0.write('\n'.encode('utf8'))
        file0.write('\n'.encode('utf8'))

        print 'Trénování a testování kmeans s maticí vytvořenou spojením TFIDF a doc2vec matic'
        maticePouzVah = maticeDoc2VecVah
        #Ncomponents = int(math.pow(len(textyPracovni), (1.0 / (1 + (np.log(len(textyPracovni)) / 10.0)))))
        P, R, Plsa, Rlsa = [], [], [], []
        for ii in range(2):
            vysledekClusteringu, clustering = UdelejKmeans2maticeTS(testSet + 'tfidfadoc2vec' + str(alphaa), soubAslozky, maxIter, tolerancee, nInit, Ncomponents, tfidfMat, maticeDoc2VecVah, tfidfMatTest, maticeDoc2VecVahTest,ii)

            Pp, Rr = evaluateResults(soubAslozky, vysledekClusteringu, nazvySoub)
            P.append(Pp)
            R.append(Rr)
            # print Pp, Rr, Pplsa, Rrlsa

        Ppom = sum(P) / float(len(P))
        if Ppom > maxP:
            maxP = Ppom
            maxAlf = alphaa
        print sum(P) / float(len(P)), sum(R) / float(len(R))
        file0.write(u'Výsledky K-means s maticí vytvořenou spojením matic tfidf a doc2vec s počtem příznaků '.encode('utf8') + str(2*Ncomponents).encode(
            'utf8') + u' na datech '.encode('utf8') + vstupPrac.encode('utf8') + u' s hodnotami alpha a minalpha: '.encode(
            'utf8') + str(alphaa).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Precision je: '.encode('utf8') + str(sum(P) / float(len(P))).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Recall je: '.encode('utf8') + str(sum(R) / float(len(R))).encode('utf8'))
        file0.write('\n'.encode('utf8'))
        file0.write('\n'.encode('utf8'))

file0.close()