# -*- coding: utf-8 -*-
# coding: utf-8

from predpriprava import *
from kmeansMethods import UdelejKmeans, UdelejKmeans2matice
import numpy as np
from evaluation import evaluateResults
from svmClass import *
from LSTMneu import *
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from statistika import *

import scipy

velikostSlovniku = 10000
#jazyk = 'english'
#vstup = 'VstupREUTERS'
#vstup = 'Vstup3raw'
#vstup = 'Vstup3rawPorov'
#vstup = "Vstup3rawTrain"
#vstup = 'Vstup3raw10NG'
#vstup = 'Vstup3rawSmall10Multi'
#vstup = 'Vstup3rawSmall10Multi1'
#vstup = 'Vstup3rawSmall10Multi2'
#vstup = 'Vstup3rawSmall5Multi'
#vstup = 'Vstup3rawSmall5Multi1'
#vstup = 'Vstup3rawSmall5Multi2'
#vstup = 'Vstup3rawSmall2Multi'
#vstup = 'Vstup3rawSmall2Multi1'
#vstup = 'Vstup3rawSmall2Multi2'
#


jazyk = 'czech'
vstup = 'VstupPrepisy'
#vstup = 'VstupPrepisyVelke'
#vstup = 'VstupPrepisyStereo'
#vstup = 'VstupPrepisyStereoVelke'
#vstup = 'VstupPrepisyStereoAmono'
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

SVMANO, PaR = 0, 0
LSTMNEU = 0
predTrain = 0
KMEANSJO = 1
vypisPr = 1
#clVyp = '1001424.txt'
clVyp = '0511'

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

#VstupProNeuNLI(soubAtextyRaw, vstup, jazyk, soubAslozky, u'_mono')


# tady se nastaví s čím se bude dále pracovat jestli s cistými texty nebo jejich lemmaty nebo tagy
'''
vstupPrac = vstup + 'CistyText'
textyPracovni = vycisteneTexty
lemmaTexty = []
tagsTexty = []
'''
vstupPrac = vstup +'Lemma'
textyPracovni = lemmaTexty
#vycisteneTexty = []
tagsTexty = []
'''
vstupPrac = vstup+'Tags'
textyPracovni = tagsTexty
vycisteneTexty = []
lemmaTexty = []
velikostSlovniku = 2000
'''

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
velikost, okno, alphaa, minalphaa, minimalniPocetCetnostiSlov = 300, 0.01, 0.019, 0.019, 5
prvniItr = 1
maxP, maxAlf, maxPref, maxAlfRef = 0.0, 0.0, 0.0, 0.0

for abc in range(1):
    if predTrain == 1:
        maticeDoc2VecVah = VytvorReprDoc2VecPredemTrenovanou(vstupPrac, textyPracovni, nazvySoub, velikost, okno, alphaa, minalphaa,
                                             minimalniPocetCetnostiSlov)
        maticeWord2VecVah = VytvorReprWord2VecPredemTrenovanou(vstupPrac, textyPracovni, nazvySoub, velikost, okno, alphaa, minalphaa,
                                               minimalniPocetCetnostiSlov, slovnik)
    else:
        maticeDoc2VecVah = VytvorReprDoc2Vec(vstupPrac, textyPracovni, nazvySoub, velikost, okno, alphaa, minalphaa, minimalniPocetCetnostiSlov)
        #maticeWord2VecVah = VytvorReprWord2Vec(vstupPrac, textyPracovni, nazvySoub, velikost, okno, alphaa, minalphaa, minimalniPocetCetnostiSlov, slovnik)
        maticeWord2VecVah = []
    if vypisPr == 1:
        print maticeDoc2VecVah[pozz][0:10]
        file1.write(str(maticeDoc2VecVah[pozz][0:10]).encode('utf8'))
        file1.write(u'\n'.encode('utf8'))
        file1.write(u'\n'.encode('utf8'))
        file1.close()
        print
    # nastavení parametrů kmeans a provedení
    maticePouzVah = tfidfMat
    maxIter = 1000000
    tolerancee = 0.00001
    nInit = 100
    #Ncomponents = int(math.pow(len(textyPracovni), (1.0 / (1 + (np.log(len(textyPracovni)) / 10.0)))))
    Ncomponents = 200

    if SVMANO == 1:
        if vstup == "Vstup3rawPorovTrain":
            UdelejSVMTRAINTEST(tfidfMat, maticeDoc2VecVah, soubAslozky, nazvySoub, Ncomponents, vstupPrac, PaR, slovnik, slovnikPole)
        elif vstup == "Vstup3rawTrain":
            UdelejSVMTRAINTEST2(tfidfMat, maticeDoc2VecVah, soubAslozky, nazvySoub, Ncomponents, vstupPrac, PaR, slovnik,
                               slovnikPole)
        else:
            UdelejSVM(tfidfMat, maticeDoc2VecVah, maticeWord2VecVah, soubAslozky, nazvySoub, Ncomponents, vstupPrac, PaR)

    if LSTMNEU == 1:
        LSTMclustering(tfidfMat, maticeDoc2VecVah, maticeWord2VecVah, soubAslozky, nazvySoub, Ncomponents, vstupPrac)
    if KMEANSJO == 1:
        documents = []
        for soubb in nazvySoub:
            documents.append(u' '.join(textyPracovni[soubb]))

        tf_vectorizer = CountVectorizer(max_df=1.0, min_df=1, max_features=10000)
        tf = tf_vectorizer.fit_transform(documents)
        tf_feature_names = tf_vectorizer.get_feature_names()

        no_topics = 20
        maxHod = 0.0
        hodOffset = 0.0
        for i in range(200):
            lda_model = LatentDirichletAllocation(n_topics=no_topics, max_iter=15, learning_method='online',
                                                  learning_offset=float(i), random_state=1)
            lda_W = lda_model.fit_transform(tf)
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

            #Pp, Rr = evaluateResults(soubAslozky, vysledkyLDA, nazvySoub)
            acc = evaluateResultsAcc2(soubAslozky, vysledkyLDA, nazvySoub)

            if maxHod < acc:
                maxHod = acc
                hodOffset = i
            print i, hodOffset
            print acc, maxHod  # , Pp, Rr
            print

        print 'Provedeno lda.'
        print hodOffset, maxHod


        lda_model = LatentDirichletAllocation(n_topics=no_topics, max_iter=15, learning_method='online',
                                              learning_offset=hodOffset, random_state=1)
        lda_W = lda_model.fit_transform(tf)
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
        statCeho = vstupPrac + u'LDA'
        statistikaVytvorenychShluku(soubAslozky, vysledkyLDA, nazvySoub, statCeho)
        acc = evaluateResultsAcc2(soubAslozky, vysledkyLDA, nazvySoub)
        print acc

        file0 = file(u'Výsledky' + vstupPrac, 'w')
        file0.write(codecs.BOM_UTF8)
        if prvniItr == 1:
            print 'Trénování a testování kmeans s TFIDF maticí a sníženou dimenzí na 2000.'
            P, R, Plsa, Rlsa = [], [], [], []
            for ii in range(10):
                vysledekClusteringu, vysledekClusteringuLSA, clustering, clusteringLSA = UdelejKmeans(vstupPrac+'TFIDF', soubAslozky, maxIter, tolerancee, nInit, 2000, maticePouzVah, ii)


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


                Pp, Rr = evaluateResults(soubAslozky, vysledekClusteringu, nazvySoub)
                Pplsa, Rrlsa = evaluateResults(soubAslozky, vysledekClusteringuLSA, nazvySoub)
                P.append(Pp)
                R.append(Rr)
                Plsa.append(Pplsa)
                Rlsa.append(Rrlsa)
                #print Pp, Rr, Pplsa, Rrlsa


            print sum(P) / float(len(P)), sum(R) / float(len(R))
            print sum(Plsa) / float(len(Plsa)), sum(Rlsa) / float(len(Rlsa))

            # nastavení parametrů kmeans a provedení
            file0.write(u'Výsledky K-means s TF-IDF maticí na datech '.encode('utf8') + vstupPrac.encode('utf8'))
            file0.write(u'\n'.encode('utf8'))
            file0.write(u'Precision je: '.encode('utf8') + str(sum(P) / float(len(P))).encode('utf8'))
            file0.write(u'\n'.encode('utf8'))
            file0.write(u'Recall je: '.encode('utf8') + str(sum(R) / float(len(R))).encode('utf8'))
            file0.write('\n'.encode('utf8'))
            file0.write(u'Výsledky K-means s TF-IDF maticí a sníženou dimenzí '.encode('utf8') + str(2000).encode('utf8') + u' na na datech '.encode('utf8') + vstupPrac.encode('utf8'))
            file0.write(u'\n'.encode('utf8'))
            file0.write(u'Precision je: '.encode('utf8') + str(sum(Plsa) / float(len(Plsa))).encode('utf8'))
            file0.write(u'\n'.encode('utf8'))
            file0.write(u'Recall je: '.encode('utf8') + str(sum(Rlsa) / float(len(Rlsa))).encode('utf8'))
            file0.write('\n'.encode('utf8'))
            file0.write('\n'.encode('utf8'))

            print 'Trénování a testování kmeans s TFIDF maticí.'
            P, R, Plsa, Rlsa = [], [], [], []
            for ii in range(10):
                vysledekClusteringu, vysledekClusteringuLSA, clustering, clusteringLSA = UdelejKmeans(vstupPrac + 'TFIDF',
                                                                                                      soubAslozky, maxIter,
                                                                                                      tolerancee, nInit,
                                                                                                      Ncomponents,
                                                                                                      maticePouzVah, ii)

                Pp, Rr = evaluateResults(soubAslozky, vysledekClusteringu, nazvySoub)
                Pplsa, Rrlsa = evaluateResults(soubAslozky, vysledekClusteringuLSA, nazvySoub)
                P.append(Pp)
                R.append(Rr)
                Plsa.append(Pplsa)
                Rlsa.append(Rrlsa)
                # print Pp, Rr, Pplsa, Rrlsa
                statCeho = vstupPrac + u'K-meansTFIDF' + str(ii)
                statistikaVytvorenychShluku(soubAslozky, vysledekClusteringu, nazvySoub, statCeho)
                statCeho = vstupPrac + u'K-meansLSATFIDF' + str(ii)
                statistikaVytvorenychShluku(soubAslozky, vysledekClusteringuLSA, nazvySoub, statCeho)

            print sum(P) / float(len(P)), sum(R) / float(len(R))
            print sum(Plsa) / float(len(Plsa)), sum(Rlsa) / float(len(Rlsa))

            # nastavení parametrů kmeans a provedení
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
        for ii in range(10):
            vysledekClusteringu, vysledekClusteringuLSA, clustering, clusteringLSA = UdelejKmeans(vstupPrac+'doc2vec'+str(alphaa), soubAslozky, maxIter, tolerancee, nInit, Ncomponents, maticePouzVah, ii)

            Pp, Rr = evaluateResults(soubAslozky, vysledekClusteringu, nazvySoub)
            Pplsa, Rrlsa = evaluateResults(soubAslozky, vysledekClusteringuLSA, nazvySoub)
            P.append(Pp)
            R.append(Rr)
            Plsa.append(Pplsa)
            Rlsa.append(Rrlsa)
            #print Pp, Rr, Pplsa, Rrlsa
            statCeho = vstupPrac + u'K-meansdoc2vec' + str(ii)
            statistikaVytvorenychShluku(soubAslozky, vysledekClusteringu, nazvySoub, statCeho)
            statCeho = vstupPrac + u'K-meansLSAdoc2vec' + str(ii)
            statistikaVytvorenychShluku(soubAslozky, vysledekClusteringuLSA, nazvySoub, statCeho)

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
        for ii in range(10):
            vysledekClusteringu, clustering = UdelejKmeans2matice(vstupPrac + 'tfidfadoc2vec' + str(alphaa), soubAslozky, maxIter, tolerancee, nInit, Ncomponents, tfidfMat, maticeDoc2VecVah,ii)

            Pp, Rr = evaluateResults(soubAslozky, vysledekClusteringu, nazvySoub)
            P.append(Pp)
            R.append(Rr)
            # print Pp, Rr, Pplsa, Rrlsa
            statCeho = u'K-meansdocTFIDF2vec' + str(ii)
            statistikaVytvorenychShluku(soubAslozky, vysledekClusteringu, nazvySoub, statCeho)

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

        alphaa += 0.001
        minalphaa += 0.001


        print maxP, maxAlf
        print maxPref, maxAlfRef

        '''
        file0.write(u'Maximum P: '.encode('utf8') + str(maxP).encode('utf8') + u' bylo dosaženo s hodnotou alpha: '.encode('utf8') + str(maxAlf).encode('utf8'))
        file0.write('\n'.encode('utf8'))
        file0.write(u'Maximum P: '.encode('utf8') + str(maxPref).encode('utf8') + u' s refinement alg. bylo dosaženo s hodnotou alpha: '.encode('utf8') + str(maxAlfRef).encode('utf8'))
        '''
        file0.close()