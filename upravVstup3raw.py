# -*- coding: utf-8 -*-
# coding: utf-8

from predpriprava import *
import os
import shutil
vstup = "Vstup3raw"

print 'Načítání souborů složky ' + vstup + ' a jejich uložení do jednotlivých souborů. '
hlSb = '*'
souboryPom, slozkyPom = ZiskejNazvySouboru(vstup + '/', hlSb)

vstupTrain = "Vstup3rawTrain/"
vstupTest = "Vstup3Test/"
pocetTrain = 13000
pocetS = 0
uzJe = {}
while pocetS < len(souboryPom)+1:
    cisloS = random.randint(0, len(souboryPom)-1)
    if not uzJe.has_key(cisloS):
        uzJe[cisloS] = cisloS
        if pocetTrain > pocetS:
            directory = vstupTrain + slozkyPom[cisloS][slozkyPom[cisloS].find("/")+1:len(slozkyPom[cisloS])]
            if not os.path.exists(directory):
                os.makedirs(directory)
            shutil.copy2(slozkyPom[cisloS] + "/" + souboryPom[cisloS], directory + "/" + souboryPom[cisloS])
        else:
            directory = vstupTest + slozkyPom[cisloS][slozkyPom[cisloS].find("/") + 1:len(slozkyPom[cisloS])]
            if not os.path.exists(directory):
                os.makedirs(directory)
            shutil.copy2(slozkyPom[cisloS] + "/" + souboryPom[cisloS], directory + "/" + souboryPom[cisloS])
        pocetS += 1
        print pocetS
