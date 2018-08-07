# -*- coding: utf-8 -*-
# coding: utf-8

from predpriprava import *
from kmeansMethods import UdelejKmeans, UdelejKmeans2matice
import numpy as np
from evaluation import evaluateResults
from svmClass import *

import scipy

velikostSlovniku = 5000
jazyk = 'english'
vstup = 'Vstup3raw'

soubAtextyRaw, soubAslozky = NacteniRawVstupu(vstup)
vycisteneTexty, lemmaTexty, tagsTexty = UpravAVysictiTexty(soubAtextyRaw, vstup, jazyk)

VstupProNeuNLI(soubAtextyRaw, vstup, jazyk, soubAslozky)
# tady se nastaví s čím se bude dále pracovat jestli s cistými texty nebo jejich lemmaty nebo tagy

vstupPrac = vstup + 'CistyText'
textyPracovni = vycisteneTexty
lemmaTexty = []
tagsTexty = []


slovnik, slovnikPole = VytvorVocab(vstupPrac, velikostSlovniku, textyPracovni)
print slovnikPole
print len(textyPracovni), type(textyPracovni)

tfidfMat, nazvySoub = vytvorTFIDF(vstupPrac, textyPracovni, slovnikPole)

file0 = file(u'Výsledky' + vstupPrac, 'w')
file0.write(codecs.BOM_UTF8)

#velikost, okno, alphaa = 5000, 10, 0.025
velikost, okno, alphaa, minalphaa, minimalniPocetCetnostiSlov = 5000, 0.01, 0.019, 0.019, 5
prvniItr = 1
maxP, maxAlf, maxPref, maxAlfRef = 0.0, 0.0, 0.0, 0.0

maticeDoc2VecVah = VytvorReprDoc2Vec(vstupPrac, textyPracovni, nazvySoub, velikost, okno, alphaa, minalphaa, minimalniPocetCetnostiSlov)
maticeWord2VecVah = VytvorReprWord2Vec(vstupPrac, textyPracovni, nazvySoub, velikost, okno, alphaa, minalphaa, minimalniPocetCetnostiSlov, slovnik)




file0.close()