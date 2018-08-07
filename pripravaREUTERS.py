# -*- coding: utf-8 -*-
# coding: utf-8

from predpriprava import *

kdeHledat = 'VstupREUTERS'
hlSb = kdeHledat + '.p'
souboryPS, slozkyPS = ZiskejNazvySouboru('PomocneSoubory/', hlSb)

if len(souboryPS) == 0:
    print 'Načítání souborů, jejich názvů a originálních shluků složky ' + kdeHledat + ' a uložení. '
    hlSb = '*'
    souboryPom, slozkyPom = ZiskejNazvySouboru(kdeHledat + '/', hlSb)

    soubKon = {}
    konf = 0
    for sb in souboryPom:
        if soubKon.has_key(sb):
            konf = 1
            break
        else:
            soubKon[sb] = sb
    if konf == 1:
        print 'Nastal konflikt s názvy souborů (stejné názvy), začíná proces přejmenování souborů aby se tento konflikt vyřešil. '
        for i in range(len(souboryPom)):
            os.rename(slozkyPom[i] + '/' + souboryPom[i], slozkyPom[i] + '/' + str(00) + str(i))
        souboryPom, slozkyPom = ZiskejNazvySouboru(kdeHledat + '/', hlSb)

    soubory = {}
    souboryText = {}
    for i in range(len(slozkyPom)):
        sl = slozkyPom[i]
        poz = sl.rfind('/') + 1
        if kdeHledat == 'VstupREUTERS':
            patTitle = r"<TITLE>(.*?)</TITLE>"
            patID = r'NEWID="(.*?)">'
            patText = r'<BODY>(.*?)</BODY>'
            zacatek = u'<REUTERS'
            konec = u'</REUTERS>'
            clanekPP = u''
            parsuj = 0
            fileS = file(sl + '/' + souboryPom[i], "r")
            for radka in fileS:
                radkaPom = (replace_nonsense_characters(unicode(radka.decode(coding_guess(radka))))).strip()
                najdiZacatek = re.findall(zacatek,radkaPom)
                najdiKonec = re.findall(konec,radkaPom)
                if parsuj == 0 and len(najdiZacatek) > 0:
                    parsuj = 1
                if parsuj == 1:
                    clanekPP += u' ' + radkaPom
                if parsuj == 1 and len(najdiKonec) > 0:
                    parsuj = 0
                    #title = re.findall(patTitle, clanekPP)[0]
                    id = re.findall(patID, clanekPP)[0]
                    jeText = (re.findall(patText, clanekPP))
                    if not len(jeText) == 0:
                        text = jeText[0]
                        souboryText[id] = text.strip()
                        soubory[id] = souboryPom[i]
                    clanekPP = u''
            fileS.close()
        else:
            soubory[souboryPom[i]] = sl[poz:len(sl)]
            clanekPP = u''
            fileS = file(sl + '/' + souboryPom[i], "r")
            for radka in fileS:
                clanekPP += u' ' + (replace_nonsense_characters(unicode(radka.decode(coding_guess(radka)))))
            fileS.close()
            souboryText[souboryPom[i]] = clanekPP.strip()

    pickle.dump([souboryText, soubory], open('PomocneSoubory/' + kdeHledat + '.p', "wb"))
else:
    print 'Načítání souborů, jejich názvů a originálních shluků složky ' + kdeHledat + ' z již předem připraveného souboru. '
    souboryText, soubory = pickle.load(open('PomocneSoubory/' + kdeHledat + '.p', "rb"))