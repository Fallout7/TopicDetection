# -*- coding: utf-8 -*-
# coding: utf-8

from predpriprava import *

def statistikaVytvorenychShluku(souboryVstup, vysledky, vysledkyNazvySoub, statistikaCeho):
    pocetVytvorenychShluku = 0.0
    pocetOriginalnichShluku = 0.0

    slozkyNazvy = {}
    for soubb in souboryVstup:
        if not slozkyNazvy.has_key(souboryVstup[soubb]):
            slozkyNazvy[souboryVstup[soubb]] = [soubb]
        else:
            slozkyNazvy[souboryVstup[soubb]] = slozkyNazvy[souboryVstup[soubb]] + [soubb]
    pocetOriginalnichShluku = len(slozkyNazvy)

    slozkyVysNazvy = {}
    for ii in range(len(vysledkyNazvySoub)):
        if not slozkyVysNazvy.has_key(vysledky[ii]):
            slozkyVysNazvy[vysledky[ii]] = [vysledkyNazvySoub[ii]]
        else:
            slozkyVysNazvy[vysledky[ii]] = slozkyVysNazvy[vysledky[ii]] + [vysledkyNazvySoub[ii]]

    pocetVytvorenychShluku = len(slozkyVysNazvy)
    if not len(slozkyNazvy) == len(slozkyVysNazvy):
        for i in range(len(slozkyNazvy)):
            if not slozkyVysNazvy.has_key(i):
                slozkyVysNazvy[i] = []

    odpovidajicShl = {}
    pocetStejnychVShl = {}
    pocetSpatneZaraz = {}
    pouzOrig = {}
    pouzVys = {}
    for i in range(len(slozkyNazvy)):
        maxxStn = 0.0
        spatnezaraz = 0.0
        odpShlOrig = ""
        odpShlVys = ""
        for shlOrig in slozkyNazvy:
            soubOrig = slozkyNazvy[shlOrig]
            for shlVys in slozkyVysNazvy:
                soubVys = slozkyVysNazvy[shlVys]
                poccStn = 0.0
                for souborVys in soubVys:
                    for souborOrig in soubOrig:
                        if souborVys == souborOrig:
                            poccStn += 1.0
                if maxxStn < poccStn:
                    if not pouzOrig.has_key(shlOrig) and not pouzVys.has_key(shlVys):
                        maxxStn = poccStn
                        spatnezaraz = len(soubOrig) - maxxStn
                        odpShlOrig = shlOrig
                        odpShlVys = shlVys
        if not odpShlVys == "":
            pouzVys[odpShlVys] = odpShlVys
            pouzOrig[odpShlOrig] = odpShlOrig
            odpovidajicShl[odpShlVys] = odpShlOrig
            pocetStejnychVShl[odpShlVys] = maxxStn
            pocetSpatneZaraz[odpShlVys] = spatnezaraz
    pocetZarSouboru = {}
    for shl in odpovidajicShl:
        pocetZarSouboru[shl] = len(slozkyVysNazvy[shl])

    pocetSoubVOrigShl = {}
    for shl in slozkyNazvy:
        pocetSoubVOrigShl[shl] = len(slozkyNazvy[shl])

    pickle.dump([pocetOriginalnichShluku, pocetVytvorenychShluku, odpovidajicShl, pocetStejnychVShl, pocetSpatneZaraz, pocetZarSouboru, pocetSoubVOrigShl, slozkyNazvy],
                open('StatistikyVysledku/Statistika' + statistikaCeho + u'.p', "wb"))


def KonecnaStatistikaVsechCyklu(vstupp):
    hlSb = u"Statistika" + vstupp + u"*" + '.p'
    souboryPS, slozkyPS = ZiskejNazvySouboru('StatistikyVysledku/', hlSb)
    pocOrigShlPr, pocVytShlPr, prvni = 0.0, 0.0, 1
    origShlStat = {}
    pocSoub = float(len(souboryPS))
    for souborr in souboryPS:
        vpocetOriginalnichShluku, pocetVytvorenychShluku, odpovidajicShl, pocetStejnychVShl, pocetSpatneZaraz, pocetZarSouboru, pocetSoubVOrigShl, slozkyNazvy = pickle.load(
            open('StatistikyVysledku/' + souborr, "rb"))
        if prvni == 1:
            for shl in slozkyNazvy:
                if not origShlStat.has_key(shl):
                    origShlStat[shl] = {"prumPocVytrShlukuPrirazenehoKorig": 0.0, "pocSprvZar": 0.0, "pocSpatZar": 0.0, "pocOrigSoubpr": 0.0}
            prvni = 0

        pocOrigShlPr = (pocOrigShlPr + vpocetOriginalnichShluku)
        pocVytShlPr = (pocVytShlPr + pocetVytvorenychShluku)
        for shl in origShlStat:
            for vytShl in odpovidajicShl:
                if shl == odpovidajicShl[vytShl]:
                    origShlStat[shl]["prumPocVytrShlukuPrirazenehoKorig"] = origShlStat[shl]["prumPocVytrShlukuPrirazenehoKorig"] + pocetZarSouboru[vytShl]
                    origShlStat[shl]["pocSprvZar"] = origShlStat[shl]["pocSprvZar"] + pocetStejnychVShl[vytShl]
                    origShlStat[shl]["pocSpatZar"] = origShlStat[shl]["pocSpatZar"] + pocetSpatneZaraz[vytShl]
            if pocetSoubVOrigShl.has_key(shl):
                origShlStat[shl]["pocOrigSoubpr"] = origShlStat[shl]["pocOrigSoubpr"] + pocetSoubVOrigShl[shl]
            else:
                origShlStat[shl]["pocOrigSoubpr"] = origShlStat[shl]["pocOrigSoubpr"] + 0.0

    pocOrigShlPr = (pocOrigShlPr / pocSoub)
    pocVytShlPr = (pocVytShlPr / pocSoub)
    for shl in origShlStat:
        origShlStat[shl]["prumPocVytrShlukuPrirazenehoKorig"] = origShlStat[shl]["prumPocVytrShlukuPrirazenehoKorig"] / pocSoub
        origShlStat[shl]["pocSprvZar"] = origShlStat[shl]["pocSprvZar"] / pocSoub
        origShlStat[shl]["pocSpatZar"] = origShlStat[shl]["pocSpatZar"] / pocSoub
        origShlStat[shl]["pocOrigSoubpr"] = origShlStat[shl]["pocOrigSoubpr"] / pocSoub

    file0 = file(u'StatistikyVysledku/Statistika' + vstupp + u'.txt', 'w')
    file0.write(codecs.BOM_UTF8)

    file0.write(u'Statistika klasifikace souborů '.encode('utf8') + vstupp.encode('utf8'))
    file0.write(u'\n'.encode('utf8'))

    # print "K původnímu počtu shluků: " + str(pocetOriginalnichShluku) + " bylo vytvořeno algoritmem " + str(pocetVytvorenychShluku) + " shluků."
    file0.write(u"K původnímu počtu shluků: ".encode('utf8') + str(pocOrigShlPr).encode(
        'utf8') + u" bylo vytvořeno algoritmem (při použití techniky 10-fold cross validation) v průměru ".encode('utf8') + str(pocVytShlPr).encode(
        'utf8') + u" shluků.".encode('utf8'))
    file0.write(u'\n'.encode('utf8'))
    for shl in origShlStat:
        file0.write(u"K originálnímu shluku: ".encode('utf8') + str(shl).encode('utf8') + u" o průměrném počtu souborů " + str(origShlStat[shl]["pocOrigSoubpr"]) + u" bylo v odpovídajících vytvořených shlucích průměrně přiřazeno: ".encode(
            'utf8') + str(origShlStat[shl]["prumPocVytrShlukuPrirazenehoKorig"]).encode('utf8') + u" souborů ".encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u"         Průměrný počet shodných souborů: ".encode('utf8') + str(origShlStat[shl]["pocSprvZar"]).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))
        file0.write(u"         průměrný počet špatně zařazených souborů: ".encode('utf8') + str(origShlStat[shl]["pocSpatZar"]).encode('utf8'))
        file0.write(u'\n'.encode('utf8'))


