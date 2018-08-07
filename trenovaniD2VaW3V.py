# -*- coding: utf-8 -*-
# coding: utf-8

from predpriprava import *
from kmeansMethods import UdelejKmeans, UdelejKmeans2matice
import numpy as np
from evaluation import evaluateResults
from upravaVstupu import *
from svmClass import *



velikostSlovniku = 5000

jazyk = 'czech'
vstup = 'TrainCZ'

#PredelejTXTdoSouboru(vstup)

hlSb = '*.p'
souboryPom, slozkyPom = ZiskejNazvySouboru(vstup + '/', hlSb)
veskereSoubory = {}
for soubb in souboryPom:
    souboryText, soubory = pickle.load(open(vstup + '/' + soubb, "rb"))
    #veskereSoubory.update(soubory)
    UpravAVysictiTextyTRAINCZ(souboryText, vstup + soubb[0:soubb.find('.')], jazyk)

hlSb = '*.p'
souboryPom, slozkyPom = ZiskejNazvySouboru('PomSoubTrainCZ/', hlSb)
vycisteneTextyVse = {}
nazvySoub = []
for soubb in souboryPom:
    if not soubb == 'TrainCZCNOVycistene.p':
        vycistene, pocet = pickle.load(open('PomSoubTrainCZ/' + soubb, "rb"))
        vycisteneTextyVse.update(vycistene)
        for soubor in vycistene:
            nazvySoub.append(soubor)
        print len(vycisteneTextyVse)
vycistene = []
print len(vycisteneTextyVse)

vstupPrac = vstup + 'CistyText'
velikost, okno, alphaa, minalphaa, minimalniPocetCetnostiSlov = 5000, 0.01, 0.019, 0.019, 5

#------------------------- vytvoření doc2vec modelu ----------------------------------------

hlSb = vstup + 'doc2vecModel.model'
souboryPS, slozkyPS = ZiskejNazvySouboru('PomSoubTrainCZ/', hlSb)

if len(souboryPS) == 0:
    print 'Trénování doc2vec modelu na češtinu.'
    train = []
    for i in range(len(nazvySoub)):
        train.append(TaggedDocument(vycisteneTextyVse[nazvySoub[i]], [nazvySoub[i]]))
        vycisteneTextyVse.pop(nazvySoub[i])
        print i, len(vycisteneTextyVse)
    nazvySoub = []
    print 'Vytvořená trénovací data.'
    model = Doc2Vec(dbow_words=0, dm_concat=0, dm_mean=0, hs=0, negative=5, iter=20, sample=0.0, size=velikost, dm=2, window=okno, min_count=minimalniPocetCetnostiSlov, workers=8, alpha=alphaa, min_alpha=minalphaa)
    #it = LabeledLineSentence(dokumentyPrac, labelss)
    print 'Vytváření slovníku.'
    model.build_vocab(train)
    print 'Trénování.'
    model.train(train)
    model.save('PomSoubTrainCZ/' + hlSb)

'''
else:
    print 'Načítání doc2vec modelu vstupu: ' + vstup
    modelD2V = Doc2Vec.load('PomSoubTrainCZ/' + hlSb)
'''

'''
#------------------------- vytvoření word2vec modelu ----------------------------------------

hlSb = vstup + 'word2vecModel.model'
souboryPS, slozkyPS = ZiskejNazvySouboru('PomSoubTrainCZ/', hlSb)

if len(souboryPS) == 0:
    print 'Vytváření word2vec modelu pro češtinu.'
    train = []
    for i in range(len(nazvySoub)):
        train.append(vycisteneTextyVse[nazvySoub[i]])

    modelW2V = Word2Vec(train, size=velikost, window=5, min_count=minimalniPocetCetnostiSlov, workers=8, alpha=alphaa)
    modelW2V.save('PomSoubTrainCZ/' + hlSb)
    
'''
'''
else:
    print 'Načítání word2vec modelu češtiny.'
    modelW2V = Word2vec.load('PomSoubTrainCZ/' + hlSb)
'''