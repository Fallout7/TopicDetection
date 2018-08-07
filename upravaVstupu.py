# -*- coding: utf-8 -*-
# coding: utf-8

from predpriprava import *


def PredelejTXTdoSouboru(vstup):
    print 'Načítání souborů složky ' + vstup + ' a jejich uložení do jednotlivých souborů. '
    hlSb = '*'
    souboryPom, slozkyPom = ZiskejNazvySouboru(vstup + '/', hlSb)

    slozky = {}
    for i in range(len(slozkyPom)):
        sloz = slozkyPom[i][slozkyPom[i].find('/') + 1:len(slozkyPom[i])]
        if not slozky.has_key(sloz):
            slozky[sloz] = [souboryPom[i]]
        else:
            slozky[sloz] = slozky[sloz] + [souboryPom[i]]

    cisloSouboru = 0
    pouziteSoubory = {}
    for slozka in slozky:
        print "Probíhá načítání článků a uložení do jediného souboru složky: " + slozka
        textyOriginal = {}
        soubory = slozky[slozka]
        for soubor in soubory:
            clanek = u''
            fileS = file(vstup + '/' + slozka + '/' + soubor, "r")
            for radka in fileS:
                radkaPom = (replace_nonsense_characters(unicode(radka.decode(coding_guess(radka))))).strip()
                clanek += u' ' + radkaPom
            fileS.close()
            if not pouziteSoubory.has_key(soubor):
                pouziteSoubory[soubor] = slozka
                textyOriginal[soubor] = clanek
            else:
                konec = 0
                while konec == 0:
                    if not pouziteSoubory.has_key(cisloSouboru):
                        pouziteSoubory[cisloSouboru] = slozka
                        textyOriginal[cisloSouboru] = clanek
                        konec = 1
                    cisloSouboru += 1
        print len(soubory)
        pickle.dump([textyOriginal, pouziteSoubory], open(vstup + '/' + slozka + '.p', "wb"))

def UpravAVysictiTextyTRAINCZ(souboryText, vstup, jazyk):
    hlSb = vstup + 'Vycistene.p'
    souboryPS, slozkyPS = ZiskejNazvySouboru('PomSoubTrainCZ/', hlSb)

    if len(souboryPS) == 0:
        print 'Probíhá čištění textu, vytváření lemmat a tagů, dále je vše uloženo pro ryhlejší načtení do souboru: ' + hlSb
        if jazyk == "english":
            tagger = Tagger.load('english-morphium-wsj-140407.tagger')
        else:
            tagger = Tagger.load('czech-morfflex-pdt-161115.tagger')
        vycistene, lemma, tags = {}, {}, {}
        pocet = 0
        print len(souboryText)
        for keyy in souboryText:
            if pocet % 1000 == 0:
                if not pocet == 0:
                    print pocet
            vycistene[keyy], lemma[keyy], tags[keyy] = vytvorLemmaTagsTokens(souboryText[keyy], tagger)
            pocet += 1
        pickle.dump([vycistene, pocet], open('PomSoubTrainCZ/' + hlSb, "wb"))
    else:
        print 'Texty již vyčištěné.'

