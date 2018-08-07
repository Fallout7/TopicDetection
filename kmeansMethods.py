# -*- coding: utf-8 -*-
# coding: utf-8

import pickle, random, time, scipy, math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from predpriprava import ZiskejNazvySouboru
from operator import add


# provede shlukování, dát si pozor aby vstup byl složka + jaká data se používají
def UdelejKmeansTS(vstupTest, slozky, maxIter, tolerance, nInit, Ncomponents, tfidf, tfidfMatTest, iterace):
    pocShluku = 0
    uzJeSl = {}
    for keyy in slozky:
        if not uzJeSl.has_key(slozky[keyy]):
            uzJeSl[slozky[keyy]] = slozky[keyy]
            pocShluku += 1
    if pocShluku == 1:
        pocShluku = 31
    print 'Počet shluků je nastaven na: ' + str(pocShluku)

    hlSb = vstupTest + 'Kmeans' + str(iterace) + '.p'
    souboryPS, slozkyPS = ZiskejNazvySouboru('PomocneSoubory/', hlSb)
    if len(souboryPS) == 0:
        #print 'Provádí se shlukování pomocí kmeans.'
        # část na výpočet výsledků s LSA
        svd = TruncatedSVD(algorithm='randomized', n_components=Ncomponents, n_iter=10, random_state=None,
                           tol=0.0)

        normalizer = Normalizer(norm='l2', copy=False)

        lsa = make_pipeline(svd, normalizer)
        X = lsa.fit_transform(tfidf)

        clusteringLSA = KMeans(n_clusters=pocShluku, init='k-means++', n_init=nInit, max_iter=maxIter,
                               tol=tolerance, precompute_distances='auto', verbose=0, random_state=None,
                               copy_x=True, n_jobs=1)
        clusteringLSA.fit(X, y=None)

        XT = lsa.transform(tfidfMatTest)

        vysledekClusteringuLSA = clusteringLSA.predict(XT)

        # část bez LSA
        clustering = KMeans(n_clusters=pocShluku, init='k-means++', n_init=nInit, max_iter=maxIter, tol=tolerance,
                            precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
        clustering.fit(tfidf, y=None)
        vysledekClusteringu = clustering.predict(tfidfMatTest)
        pickle.dump([vysledekClusteringu, vysledekClusteringuLSA, clustering, clusteringLSA],
                    open('PomocneSoubory/' + hlSb, "wb"))
    else:
        #print 'Načítají se výsledky shlukování pomocí kmeans.' + vstup
        vysledekClusteringu, vysledekClusteringuLSA, clustering, clusteringLSA = pickle.load(
            open('PomocneSoubory/' + hlSb, "rb"))

    return vysledekClusteringu, vysledekClusteringuLSA, clustering, clusteringLSA

# provede shlukování, dát si pozor aby vstup byl složka + jaká data se používají
def UdelejKmeans2maticeTS(vstup, slozky, maxIter, tolerance, nInit, Ncomponents, tfidf, doc2vec, tfidfTest, doc2vecTest, iterace):
    pocShluku = 0
    uzJeSl = {}
    for keyy in slozky:
        if not uzJeSl.has_key(slozky[keyy]):
            uzJeSl[slozky[keyy]] = slozky[keyy]
            pocShluku += 1
    if pocShluku == 1:
        pocShluku = 31
    print 'Počet shluků je nastaven na: ' + str(pocShluku)

    hlSb = vstup + 'KmeansTFIDFaDoc2vec' + str(iterace) + '.p'
    souboryPS, slozkyPS = ZiskejNazvySouboru('PomocneSoubory/', hlSb)
    if len(souboryPS) == 0:
        #print 'Provádí se shlukování pomocí kmeans.'
        # část na výpočet výsledků s LSA tfidf
        svd = TruncatedSVD(algorithm='randomized', n_components=Ncomponents, n_iter=10, random_state=None,
                           tol=0.0)

        normalizer = Normalizer(norm='l2', copy=False)

        lsa = make_pipeline(svd, normalizer)
        Xtfidf = lsa.fit_transform(tfidf)

        XtfidfTest = lsa.transform(tfidfTest)

        # část na výpočet výsledků s LSA doc2vec
        svd = TruncatedSVD(algorithm='randomized', n_components=Ncomponents, n_iter=10, random_state=None,
                           tol=0.0)

        normalizer = Normalizer(norm='l2', copy=False)

        lsa = make_pipeline(svd, normalizer)
        Xdoc2vec = lsa.fit_transform(doc2vec)
        # část na výpočet výsledků s LSA doc2vec
        Xdoc2vecTest = lsa.transform(doc2vecTest)

        maticeTFIDFD2V = np.append(Xtfidf, Xdoc2vec, axis=1)

        # shlukování k-means
        clustering = KMeans(n_clusters=pocShluku, init='k-means++', n_init=nInit, max_iter=maxIter, tol=tolerance,
                            precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
        clustering.fit(maticeTFIDFD2V, y=None)

        maticeTFIDFD2VTest = np.append(XtfidfTest, Xdoc2vecTest, axis=1)

        vysledekClusteringu = clustering.predict(maticeTFIDFD2VTest)
        pickle.dump([vysledekClusteringu, clustering],
                    open('PomocneSoubory/' + hlSb, "wb"))
    else:
        #print 'Načítají se výsledky shlukování pomocí kmeans.' + vstup
        vysledekClusteringu, clustering = pickle.load(open('PomocneSoubory/' + hlSb, "rb"))

    return vysledekClusteringu, clustering

# provede shlukování, dát si pozor aby vstup byl složka + jaká data se používají
def UdelejKmeans(vstup, slozky, maxIter, tolerance, nInit, Ncomponents, tfidf, iterace):
    pocShluku = 0
    uzJeSl = {}
    for keyy in slozky:
        if not uzJeSl.has_key(slozky[keyy]):
            uzJeSl[slozky[keyy]] = slozky[keyy]
            pocShluku += 1
    if pocShluku == 1:
        pocShluku = 31
    print 'Počet shluků je nastaven na: ' + str(pocShluku)

    hlSb = vstup + 'Kmeans' + str(iterace) + '.p'
    souboryPS, slozkyPS = ZiskejNazvySouboru('PomocneSoubory/', hlSb)
    if len(souboryPS) == 0:
        #print 'Provádí se shlukování pomocí kmeans.'
        # část na výpočet výsledků s LSA
        svd = TruncatedSVD(algorithm='randomized', n_components=Ncomponents, n_iter=10, random_state=None,
                           tol=0.0)

        normalizer = Normalizer(norm='l2', copy=False)

        lsa = make_pipeline(svd, normalizer)
        X = lsa.fit_transform(tfidf)

        clusteringLSA = KMeans(n_clusters=pocShluku, init='k-means++', n_init=nInit, max_iter=maxIter,
                               tol=tolerance, precompute_distances='auto', verbose=0, random_state=None,
                               copy_x=True, n_jobs=1)
        clusteringLSA.fit(X, y=None)
        vysledekClusteringuLSA = clusteringLSA.predict(X)

        # část bez LSA
        clustering = KMeans(n_clusters=pocShluku, init='k-means++', n_init=nInit, max_iter=maxIter, tol=tolerance,
                            precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
        clustering.fit(tfidf, y=None)
        vysledekClusteringu = clustering.predict(tfidf)
        pickle.dump([vysledekClusteringu, vysledekClusteringuLSA, clustering, clusteringLSA],
                    open('PomocneSoubory/' + hlSb, "wb"))
    else:
        #print 'Načítají se výsledky shlukování pomocí kmeans.' + vstup
        vysledekClusteringu, vysledekClusteringuLSA, clustering, clusteringLSA = pickle.load(
            open('PomocneSoubory/' + hlSb, "rb"))

    return vysledekClusteringu, vysledekClusteringuLSA, clustering, clusteringLSA

# provede shlukování, dát si pozor aby vstup byl složka + jaká data se používají
def UdelejKmeans10Fold(vstup, matTrain, matTest, matTranLSA, matTestLSA, maxIter, tolerance, nInit, iterace, pocShluku, crossvalStep):

    hlSb = vstup + str(crossvalStep) +'Kmeans10fold' + str(iterace) + '.p'
    souboryPS, slozkyPS = ZiskejNazvySouboru('PomocneSoubory/', hlSb)
    if len(souboryPS) == 0:
        #print 'Provádí se shlukování pomocí kmeans.'


        clusteringLSA = KMeans(n_clusters=pocShluku, init='k-means++', n_init=nInit, max_iter=maxIter,
                               tol=tolerance, precompute_distances='auto', verbose=0, random_state=None,
                               copy_x=True, n_jobs=1)
        clusteringLSA.fit(matTranLSA, y=None)
        vysledekClusteringuLSA = clusteringLSA.predict(matTestLSA)

        # část bez LSA
        clustering = KMeans(n_clusters=pocShluku, init='k-means++', n_init=nInit, max_iter=maxIter, tol=tolerance,
                            precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
        clustering.fit(matTrain, y=None)
        vysledekClusteringu = clustering.predict(matTest)
        pickle.dump([vysledekClusteringu, vysledekClusteringuLSA, clustering, clusteringLSA],
                    open('PomocneSoubory/' + hlSb, "wb"))
    else:
        #print 'Načítají se výsledky shlukování pomocí kmeans.' + vstup
        vysledekClusteringu, vysledekClusteringuLSA, clustering, clusteringLSA = pickle.load(
            open('PomocneSoubory/' + hlSb, "rb"))

    return vysledekClusteringu, vysledekClusteringuLSA, clustering, clusteringLSA

# provede shlukování, dát si pozor aby vstup byl složka + jaká data se používají
def UdelejKmeans2matice(vstup, slozky, maxIter, tolerance, nInit, Ncomponents, tfidf, doc2vec, iterace):
    pocShluku = 0
    uzJeSl = {}
    for keyy in slozky:
        if not uzJeSl.has_key(slozky[keyy]):
            uzJeSl[slozky[keyy]] = slozky[keyy]
            pocShluku += 1
    if pocShluku == 1:
        pocShluku = 31
    print 'Počet shluků je nastaven na: ' + str(pocShluku)

    hlSb = vstup + 'KmeansTFIDFaDoc2vec' + str(iterace) + '.p'
    souboryPS, slozkyPS = ZiskejNazvySouboru('PomocneSoubory/', hlSb)
    if len(souboryPS) == 0:
        #print 'Provádí se shlukování pomocí kmeans.'
        # část na výpočet výsledků s LSA tfidf
        svd = TruncatedSVD(algorithm='randomized', n_components=Ncomponents, n_iter=10, random_state=None,
                           tol=0.0)

        normalizer = Normalizer(norm='l2', copy=False)

        lsa = make_pipeline(svd, normalizer)
        Xtfidf = lsa.fit_transform(tfidf)

        # část na výpočet výsledků s LSA doc2vec
        svd = TruncatedSVD(algorithm='randomized', n_components=Ncomponents, n_iter=10, random_state=None,
                           tol=0.0)

        normalizer = Normalizer(norm='l2', copy=False)

        lsa = make_pipeline(svd, normalizer)
        Xdoc2vec = lsa.fit_transform(doc2vec)

        maticeTFIDFD2V = np.append(Xtfidf, Xdoc2vec, axis=1)
        # shlukování k-means
        clustering = KMeans(n_clusters=pocShluku, init='k-means++', n_init=nInit, max_iter=maxIter, tol=tolerance,
                            precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
        clustering.fit(maticeTFIDFD2V, y=None)

        vysledekClusteringu = clustering.predict(maticeTFIDFD2V)
        pickle.dump([vysledekClusteringu, clustering],
                    open('PomocneSoubory/' + hlSb, "wb"))
    else:
        #print 'Načítají se výsledky shlukování pomocí kmeans.' + vstup
        vysledekClusteringu, clustering = pickle.load(open('PomocneSoubory/' + hlSb, "rb"))

    return vysledekClusteringu, clustering

# provede shlukování, dát si pozor aby vstup byl složka + jaká data se používají
def UdelejKmeansRefinementAlg(vstup, slozky, maxIter, tolerance, nInit, Ncomponents, tfidf, iterace):
    pocShluku = 0
    uzJeSl = {}
    for keyy in slozky:
        if not uzJeSl.has_key(slozky[keyy]):
            uzJeSl[slozky[keyy]] = slozky[keyy]
            pocShluku += 1
    if pocShluku == 1:
        pocShluku = 31
    #print 'Počet shluků je nastaven na: ' + str(pocShluku)

    hlSb = vstup + 'Kmeans' + str(iterace) + '.p'
    souboryPS, slozkyPS = ZiskejNazvySouboru('PomocneSoubory/', hlSb)
    if len(souboryPS) == 0:
        #print 'Provádí se shlukování pomocí kmeans.'
        # část na výpočet výsledků s LSA
        svd = TruncatedSVD(algorithm='randomized', n_components=Ncomponents, n_iter=10, random_state=None,
                           tol=0.0)

        normalizer = Normalizer(norm='l2', copy=False)

        lsa = make_pipeline(svd, normalizer)
        X = scipy.sparse.csr_matrix(lsa.fit_transform(tfidf))

        J = int(X.shape[0]/150.0)
        K = pocShluku
        stredyInit = RefinementAlgorithm(X, K, maxIter, tolerance, J, 100)

        clusteringLSA = KMeans(n_clusters=pocShluku, init=stredyInit, n_init=1, max_iter=maxIter,
                               tol=tolerance, precompute_distances='auto', verbose=0, random_state=None,
                               copy_x=True, n_jobs=1)
        clusteringLSA.fit(X, y=None)
        vysledekClusteringuLSA = clusteringLSA.predict(X)

        # část bez LSA
        K = pocShluku
        stredyInit = RefinementAlgorithm(scipy.sparse.csr_matrix(tfidf), K, maxIter, tolerance, J, 100)

        clustering = KMeans(n_clusters=pocShluku, init=stredyInit, n_init=1, max_iter=maxIter, tol=tolerance,
                            precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
        clustering.fit(tfidf, y=None)
        vysledekClusteringu = clustering.predict(tfidf)
        pickle.dump([vysledekClusteringu, vysledekClusteringuLSA, clustering, clusteringLSA],
                    open('PomocneSoubory/' + hlSb, "wb"))
    else:
        #print 'Načítají se výsledky shlukování pomocí kmeans.' + vstup
        vysledekClusteringu, vysledekClusteringuLSA, clustering, clusteringLSA = pickle.load(
            open('PomocneSoubory/' + hlSb, "rb"))

    return vysledekClusteringu, vysledekClusteringuLSA, clustering, clusteringLSA

def ulozenaCastMainRinfinmentAlg(file0, prvniItr, textyPracovni, vstupPrac, tfidfMat, soubAslozky, nazvySoub, maxIter, tolerancee, nInit):
    # ------------------tady s reinfinement alg shlukování (tedy algoritmus na předpovědění centroidů)------------------------
    if prvniItr == 1:
        Ncomponents = int(math.pow(len(textyPracovni), (1.0 / (1 + (np.log(len(textyPracovni)) / 10.0)))))
        print 'Trénování a testování kmeans s TFIDF maticí a ref. alg.'
        maticePouzVah = tfidfMat
        P, R, Plsa, Rlsa = [], [], [], []
        for ii in range(10):
            vysledekClusteringu, vysledekClusteringuLSA, clustering, clusteringLSA = UdelejKmeansRefinementAlg(
                vstupPrac + 'TFIDFrefAlg', soubAslozky, maxIter, tolerancee, nInit, Ncomponents, maticePouzVah, ii)
            '''
            Pp, Rr = evaluateResults(soubAslozky, vysledekClusteringu, nazvySoub)
            Pplsa, Rrlsa = evaluateResults(soubAslozky, vysledekClusteringuLSA, nazvySoub)
            P.append(Pp)
            R.append(Rr)
            Plsa.append(Pplsa)
            Rlsa.append(Rrlsa)
            # print Pp, Rr, Pplsa, Rrlsa
            
        print sum(P) / float(len(P)), sum(R) / float(len(R))
        print sum(Plsa) / float(len(Plsa)), sum(Rlsa) / float(len(Rlsa))
        '''
        # nastavení parametrů kmeans a provedení
        maxIter = 10000
        tolerancee = 0.0001
        nInit = 100
        file0.write(
            u'Výsledky K-means s TF-IDF maticí a Reinfinement Alg. na datech '.encode('utf8') + vstupPrac.encode(
                'utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Precision je: '.encode('utf8') + str(sum(P) / float(len(P))).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Recall je: '.encode('utf8') + str(sum(R) / float(len(R))).encode('utf8'))
        file0.write('\n'.encode('utf8'))
        file0.write(u'Výsledky K-means s TF-IDF maticí, Reinfinement Alg. a sníženou dimenzí '.encode('utf8') + str(
            Ncomponents).encode('utf8') + u' na na datech '.encode('utf8') + vstupPrac.encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Precision je: '.encode('utf8') + str(sum(Plsa) / float(len(Plsa))).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u'Recall je: '.encode('utf8') + str(sum(Rlsa) / float(len(Rlsa))).encode('utf8'))
        file0.write('\n'.encode('utf8'))
        file0.write('\n'.encode('utf8'))
        prvniItr = 0

    '''
    print 'Trénování a testování kmeans s doc2vec maticí a ref. alg.'
    Ncomponents = 1000
    maticePouzVah = maticeDoc2VecVah
    P, R, Plsa, Rlsa = [], [], [], []
    for ii in range(10):
        vysledekClusteringu, vysledekClusteringuLSA, clustering, clusteringLSA = UdelejKmeansRefinementAlg(vstupPrac+'doc2vecRefAlg'+str(alphaa), soubAslozky, maxIter, tolerancee, nInit, Ncomponents, maticePouzVah, ii)
        Pp, Rr = evaluateResults(soubAslozky, vysledekClusteringu, nazvySoub)
        Pplsa, Rrlsa = evaluateResults(soubAslozky, vysledekClusteringuLSA, nazvySoub)
        P.append(Pp)
        R.append(Rr)
        Plsa.append(Pplsa)
        Rlsa.append(Rrlsa)
        #print Pp, Rr, Pplsa, Rrlsa

    Ppom = sum(P) / float(len(P))
    PpomLSa = sum(Plsa) / float(len(Plsa))
    if Ppom > maxPref:
        maxPref = Ppom
        maxAlfRef = alphaa
    if PpomLSa > maxPref:
        maxPref = PpomLSa
        maxAlfRef = alphaa
    print sum(P) / float(len(P)), sum(R) / float(len(R))
    print sum(Plsa) / float(len(Plsa)), sum(Rlsa) / float(len(Rlsa))
    file0.write(u'Výsledky K-means s doc2vec maticí a Reinfinement Alg. na datech '.encode('utf8') + vstupPrac.encode('utf8')+ u' s hodnotami alpha a minalpha: '.encode('utf8') + str(alphaa).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Precision je: '.encode('utf8') + str(sum(P) / float(len(P))).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Recall je: '.encode('utf8') + str(sum(R) / float(len(R))).encode('utf8'))
    file0.write('\n'.encode('utf8'))
    file0.write(u'Výsledky K-means s doc2vec maticí, Reinfinement Alg. a sníženou dimenzí '.encode('utf8') + str(Ncomponents).encode('utf8') + u' na na datech '.encode('utf8') + vstupPrac.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Precision je: '.encode('utf8') + str(sum(Plsa) / float(len(Plsa))).encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    file0.write(u'Recall je: '.encode('utf8') + str(sum(Rlsa) / float(len(Rlsa))).encode('utf8'))
    file0.write('\n'.encode('utf8'))
    file0.write(u'--------------------------------------------------------------------------------------------------------------------------------------'.encode('utf8'))
    file0.write('\n'.encode('utf8'))
    file0.write('\n'.encode('utf8'))
    '''

def KmeansMod(maxIter,K ,stredy,matPrac,tolerance):
    clustering = KMeans(n_clusters = K, init=stredy, n_init=1, max_iter=maxIter, tol=tolerance,
                        precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
    clustering.fit(matPrac, y=None)

    return clustering.cluster_centers_

def KmeansPred(maxIter,K ,stredy,matPrac,tolerance):
    clustering = KMeans(n_clusters=K, init=stredy, n_init=1, max_iter=maxIter, tol=tolerance,
                        precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)


    clustering.fit(matPrac, y=None)

    return clustering.cluster_centers_

def vzdalenostMan(vek1, vek2):
    return np.sqrt(np.sum(np.power(vek1-vek2,2)))

def RefinementAlgorithm(matPrac, K, maxIter, tolerance, J, kolikvJ):
    #vytvoření J podsetů Si o velikostech kolikvJ
    Si = [] #na i té pozici obsažen subset dat o velikosti kolikvJ
    pouzPoz = {}
    for i in range(J):
        pocetVyb = 0
        smallSubset = []
        while pocetVyb < kolikvJ:
            poz = random.randint(0, matPrac.shape[0] - 1)
            if not pouzPoz.has_key(poz):
                pouzPoz[poz] = poz
                smallSubset.append(matPrac[poz].toarray()[0])
                pocetVyb += 1
        Si.append(smallSubset)
    #výběr K inicializačních středů
    pocetVyb = 0
    stredy = []
    pouzPoz = {}
    while pocetVyb < K:
        poz = random.randint(0, (matPrac.shape[0]) - 1)
        if not pouzPoz.has_key(poz):
            pouzPoz[poz] = poz
            stredy.append(np.array(matPrac[poz].toarray()[0]))
            pocetVyb += 1

    stredy = np.array(stredy)
    CM = [] #středy shluků
    CMi = [] #středy i tých shuků podle i tého subsetu Si
    for i in range(len(Si)):
        CMi.append(KmeansMod(maxIter, K, stredy, Si[i], tolerance))
        if not CM == []:
            for j in range(len(CM)):
                CM[j] = np.divide(map(add, CM[j], CMi[i][j]), 2.0)
        else:
            CM = CMi[i]

    FMi = []
    FMS = []
    CM = np.array(CM)
    for i in range(len(Si)):
        FMi.append(KmeansPred(maxIter, K, CMi[i], CM, tolerance))
        if not FMS == []:
            for j in range(len(FMS)):
                FMS[j] = np.divide(map(add, FMS[j], FMi[i][j]), 2.0)
        else:
            FMS = FMi[i]

    FM = []
    hodMax = 10000000.0000
    for i in range(len(Si)):
        hodVz = []
        for j in range(len(FMi)):
            for k in range(len(FMi[j])):
                hodVz.append(vzdalenostMan(FMi[i][k], FMi[j][k]))
        hod = np.mean(hodVz)
        if hod < hodMax:
            hodMax = hod
            pozice = i

    FM = FMi[pozice]
    return FM



