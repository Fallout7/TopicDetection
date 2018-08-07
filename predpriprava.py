# -*- coding: utf-8 -*-
# coding: utf-8
import os, fnmatch, time, re, unicodedata, pickle, codecs, math, sklearn, random, shutil, csv
import numpy as np
from ufal.morphodita import *
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Doc2Vec
from gensim.models import Word2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import LabeledSentence
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier


class enc:
    def __init__(self, coding):
        self.coding = coding  # type of coding
        self.number = 0  # number of czech chars in the text


# nahrazuje  chybné znaky správnými (chybné znaky vznikají při špatném kódování)
def replace_nonsense_characters(text):
    text = re.sub(u'[”“]', u'"', text)
    text = re.sub(u"’", u"'", text)
    text = re.sub(u"[–—]", u"-", text)
    text = re.sub(u'[\x00-\x08\x0B\x0C\x0E-\x1F\xA0\u2009]+', u' ', text)
    text = re.sub(u'\t+', ' ', text)
    text = re.sub(u'  +', u' ', text)
    text = re.sub(u'( *[\r\n])+ *', u'\n', text)
    return text.strip()


# odhadne jak je daný string kódován, vrací tento odhad
def coding_guess(text):
    encodings = ('ascii', 'iso-8859-2', 'cp1250', 'cp1251', 'cp852', 'utf-8')
    czech_chars = u"áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ"

    # first selection of the coding ... codings that change text to unicode without an error
    encoding = []
    for i in encodings:
        try:
            unicode(text, i)
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        else:
            encoding.append(enc(i))  # insert into coding list

    # second selection of the coding ... this will choose coding with maximal number of czech chars
    max = -1
    for code in encoding:
        for char in unicode(text, code.coding):
            if char in czech_chars:
                code.number = code.number + 1
        if code.number > max:
            max = code.number
            max_code = code.coding
    return max_code


# zjišťuje názvy složek a souborů v dané složce (vstupem je: složka kde má hledat a typ souborů co má hledat)
# vrací jak názvy podsložek tak názvy souborů
def ZiskejNazvySouboru(treeroot, pattern):
    results = []
    filess = []
    dirrs = []
    for base, dirs, files in os.walk(treeroot):
        goodfiles = fnmatch.filter(files, pattern)
        results.extend(os.path.join(base, f) for f in goodfiles)
    for i in range(len(results)):
        (dirname, filename) = os.path.split(results[i])
        filess.append(filename)
        dirrs.append(dirname)
    return filess, dirrs



def display_topics(H, W, feature_names, documents, no_top_words):
    vysledkyLDA = []
    for topic_idx, topic in enumerate(H):
        '''
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])
        '''
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:len(documents)]
        print top_doc_indices
        print len(top_doc_indices)
        time.sleep(1)
        for doc_index in top_doc_indices:
            #print documents[doc_index]
            vysledkyLDA.append(topic_idx)
    return vysledkyLDA


def NacteniRawVstupu(kdeHledat):
    hlSb = kdeHledat + '.p'
    souboryPS, slozkyPS = ZiskejNazvySouboru('PomocneSoubory/', hlSb)
    if len(souboryPS) == 0:
        print 'Načítání souborů, jejich názvů a originálních shluků složky ' + kdeHledat + ' a uložení. '
        hlSb = '*'
        souboryPom, slozkyPom = ZiskejNazvySouboru(kdeHledat + '/', hlSb)

        print len(souboryPom), len(slozkyPom)
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
            elif not kdeHledat.find('Vstup3raw') == -1:
                soubory[souboryPom[i]] = sl[poz:len(sl)]
                clanekPP = u''
                fileS = file(sl + '/' + souboryPom[i], "r")
                zacni = 0
                for radka in fileS:
                    radka = unicode(radka.decode(coding_guess(radka)))
                    if zacni == 1:
                        clanekPP += u' ' + (replace_nonsense_characters(radka))
                    if radka == u'' or radka == u' ' or radka == u'\n':
                        zacni = 1
                fileS.close()
                souboryText[souboryPom[i]] = clanekPP.strip()

            else:
                soubory[souboryPom[i]] = sl[poz:len(sl)]
                clanekPP = u''
                fileS = file(sl + '/' + souboryPom[i], "r")
                for radka in fileS:
                    clanekPP += u' ' + (replace_nonsense_characters(unicode(radka.decode(coding_guess(radka)))))
                fileS.close()
                #print souboryPom[i]
                #print clanekPP
                souboryText[souboryPom[i]] = clanekPP.strip()

        pickle.dump([souboryText, soubory], open('PomocneSoubory/' + kdeHledat + '.p', "wb"))
    else:
        print 'Načítání souborů, jejich názvů a originálních shluků složky ' + kdeHledat + ' z již předem připraveného souboru. '
        souboryText, soubory = pickle.load(open('PomocneSoubory/' + kdeHledat + '.p', "rb"))

    return souboryText, soubory


# odstraňuje interpunkci z textu
def odstran_interpunkci(text):
    #text = re.sub(r'\W+', u' ', text)
    text = text.replace(u'...', u'')
    text = text.replace(u'\u2026', u' ')
    text = text.replace(u"„", u"")
    text = text.replace(u"”", u"")
    text = text.replace(u'"', u'')
    text = text.replace(u'.', u' ')
    text = text.replace(u',', u'')
    text = text.replace(u'!', u'')
    text = text.replace(u'-', u'')
    text = text.replace(u'/', u'')
    text = text.replace(u'{', u'')
    text = text.replace(u'}', u'')
    text = text.replace(u'<', u'')
    text = text.replace(u'>', u'')
    text = text.replace(u':', u'')
    text = text.replace(u'?', u'')
    text = text.replace(u'(', u'')
    text = text.replace(u')', u'')
    text = text.replace(u'[', u'')
    text = text.replace(u']', u'')
    text = text.replace(u'´', u' ')
    text = text.replace(u'$', u' ')
    text = text.replace(u'%', u' ')
    text = text.replace(u'#', u' ')
    text = text.replace(u'@', u' ')
    text = text.replace(u'*', u' ')
    text = text.replace(u'+', u' ')
    text = text.replace(u"'", u'')
    text = text.replace(u"^", u' ')
    text = text.replace(u'   ', u' ')
    text = text.replace(u'  ', u' ')
    text = text.replace(u'roku1980', u'roku 1980')
    return text


def odstranNesmyslyZTextu(text):
    text = text.replace(u'\u010f', u'')
    text = text.replace(u'\u0165', u'')
    text = text.replace(u'\u017c', u'')
    return text


def encode_entities(text):
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def vytvorLemmaTagsTokens(textVstup, tagger):
    forms = Forms()
    lemmas = TaggedLemmas()
    tokens = TokenRanges()
    tokenizer = tagger.newTokenizer()

    text = odstranNesmyslyZTextu(replace_nonsense_characters(textVstup))
    # vycistenyText = odstran_interpunkci(text).lower()
    maxDel = 2000
    puvodniDelka = len(text)
    textCasti = []
    if puvodniDelka > maxDel:
        od = 0
        do = 0
        hledej = 0
        for i in range(len(text)):
            if not i == 0:
                if i % maxDel == 0:
                    hledej = 1
                if hledej == 1 or i == len(text) - 1:
                    if text[i] == u'.' or text[i] == u'?' or text[i] == u'!' or i == len(text) - 1:
                        do = i + 1
                        textCasti.append(text[od:do].strip())
                        od = i + 1
                        hledej = 0
        if len(textCasti) == 1 and len(textCasti[0]) > maxDel:
            textCasti = []
            od = 0
            do = 0
            hledej = 0
            for i in range(len(text)):
                if not i == 0:
                    if i % maxDel == 0:
                        hledej = 1
                    if hledej == 1 or i == len(text) - 1:
                        if text[i] == u' ' or i == len(text) - 1:
                            do = i + 1
                            textCasti.append(text[od:do].strip())
                            od = i + 1
                            hledej = 0
    else:
        textCasti.append(text)
    #print len(text), len(textCasti)
    lemmata, tagy, tokeny = [], [], []
    for textC in textCasti:
        tokenizer.setText(textC)
        t = 0
        vysll = u''
        while tokenizer.nextSentence(forms, tokens):
            tagger.tag(forms, lemmas)
            for i in range(len(lemmas)):
                lemma = lemmas[i]
                token = tokens[i]
                lemmata.append(encode_entities(lemma.lemma))
                tagy.append(encode_entities(lemma.tag))
                tokeny.append(encode_entities(textC[token.start: token.start + token.length]))
                vysll += ('%s%s<token lemma="%s" tag="%s">%s</token>%s' % (
                    encode_entities(textC[t: token.start]),
                    "<sentence>" if i == 0 else "",
                    encode_entities(lemma.lemma),
                    encode_entities(lemma.tag),
                    encode_entities(textC[token.start: token.start + token.length]),
                    "</sentence>" if i + 1 == len(lemmas) else "",
                ))
                t = token.start + token.length
                vysll += encode_entities(textC[t:])

    lemmataUpr, tagyUpr, tokenyUpr = [], [], []
    for ij in range(len(tagy)):
        lemm = odstran_interpunkci(lemmata[ij].lower()).replace(u' ', u'')
        if not len(lemm) == 0:
            lemmataUpr.append(lemm)
            tagyUpr.append(odstran_interpunkci(tagy[ij]).replace(u' ', u''))
            tokenyUpr.append(odstran_interpunkci(tokeny[ij].lower()).replace(u' ', u''))

    # vycistenyText = u' '.join(tokenyUpr)
    # tady maximálně další úprava lemmat a tagů, prozatím vytvářeny podle originálního textu, ne předem vyčištěného
    # lemmataVseP = odstran_interpunkci(u' '.join(lemmata).lower()).split()
    # tagyVseP = odstran_interpunkci(u' '.join(tagy)).split()

    return tokenyUpr, lemmataUpr, tagyUpr


def UpravAVysictiTexty(souboryText, vstup, jazyk):
    hlSb = vstup + 'VycisLemmTag.p'
    souboryPS, slozkyPS = ZiskejNazvySouboru('PomocneSoubory/', hlSb)

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
                print pocet
            vycistene[keyy], lemma[keyy], tags[keyy] = vytvorLemmaTagsTokens(souboryText[keyy], tagger)
            pocet += 1
        pickle.dump([vycistene, lemma, tags], open('PomocneSoubory/' + hlSb, "wb"))
    else:
        print 'Načítání vyčištěného textu, lemmat a tagů ze souboru: ' + hlSb
        vycistene, lemma, tags = pickle.load(open('PomocneSoubory/' + hlSb, "rb"))
    print len(souboryText), len(vycistene), len(lemma), len(tags)
    return vycistene, lemma, tags


def VytvorVocab(vstup, pocetFeatures, texty):
    hlSb = vstup + 'Slovnik.p'
    souboryPS, slozkyPS = ZiskejNazvySouboru('PomocneSoubory/', hlSb)

    if len(souboryPS) == 0:
        print 'Vytváření slovníku: ' + hlSb
        termyVse = {}
        clankyVse = {}
        N = len(texty)
        for j in texty:
            termyCl = texty[j]
            termyClDict = {}
            for term in termyCl:
                termyClDict[term] = term
                if not termyVse.has_key(term):
                    termyVse[term] = term
            clankyVse[j] = termyClDict
        print 'bude A'
        A = {}
        for key in texty:
            pole = texty[key]
            termyCl = {}
            for slo in pole:
                if not termyCl.has_key(slo):
                    termyCl[slo] = slo
                    if A.has_key(slo):
                        A[slo] = A[slo] + 1
                    else:
                        A[slo] = 1
        print 'bude B'
        B = {}
        for term in termyVse:
            poc = 0.0
            for cl in clankyVse:
                if not clankyVse[cl].has_key(term):
                    poc += 1.0
            B[term] = poc
        print 'bude C'
        C = {}
        for cl in clankyVse:
            poc = 0.0
            for term in termyVse:
                if not clankyVse[cl].has_key(term):
                    poc += 1.0
            C[cl] = poc
        print 'vyber ' + str(pocetFeatures) + ' příznaků z celkového počtu ' + str(len(termyVse)) + ' termů.'
        maxx = [-10000000.0] * pocetFeatures
        termm = [''] * pocetFeatures
        print len(maxx)
        termy = {}
        for cl in clankyVse:
            termyClanku = clankyVse[cl]
            #for term in termyVse:
            for term in termyClanku:
                if not termy.has_key(term):
                    hod1 = (A[term] * N)
                    hod2 = A[term] + B[term]
                    hod3 = A[term] + C[cl]
                    MIt = (math.log((hod1) / ((hod3) * (hod2))))
                    minimum = min(maxx)
                    poz = maxx.index(min(maxx))
                    if MIt > minimum:
                        maxx[poz] = MIt
                        termm[poz] = term
                        termy[term] = term
        # print termm
        slovnik = {}
        slovnikPole = []
        for k in range(len(termm)):
            slovnik[termm[k]] = termm[k]
            slovnikPole.append(termm[k])

        # případ extrahování stop slov
        '''
        idf = {}
        stopSlova = {}
        stopSlovaPole = []
        # slovnikPole = []
        for slo in A:
            hod = math.log(float(len(stemClankyTrain)) / A[slo])
            idf[slo] = hod
            if hod < mezStopSlov:
                stopSlova[slo] = slo
                stopSlovaPole.append(slo)
        '''
        pickle.dump([slovnik, slovnikPole], open("PomocneSoubory/" + hlSb, "wb"))
    else:
        print 'Načítání slovníku: ' + hlSb
        slovnik, slovnikPole = pickle.load(open("PomocneSoubory/" + hlSb, "rb"))
    return slovnik, slovnikPole


def vytvorTFIDF(vstup, texty, slovnik):
    hlSb = vstup + 'TFIDF.p'
    souboryPS, slozkyPS = ZiskejNazvySouboru('PomocneSoubory/', hlSb)

    nazvySoub = []
    if len(souboryPS) == 0:
        textyPol = []
        for key in texty:
            textyPol.append(u' '.join(texty[key]))
            nazvySoub.append(key)

        print 'Vytváření matice TF-IDF vah vstupu: ' + vstup
        # tfidf matice sklearn train data ----------------------------------------------------------------------------------
        tfidf_vectorizer = TfidfVectorizer(max_df=1.0, max_features=None, min_df=0.0, vocabulary=slovnik,
                                           stop_words=None, use_idf=True, tokenizer=None, ngram_range=(1, 1),
                                           sublinear_tf=1)
        tfidf = tfidf_vectorizer.fit_transform(textyPol)

        pickle.dump([tfidf, nazvySoub], open('PomocneSoubory/' + hlSb, "wb"))
    else:
        print 'Načítání matice TF-IDF vah vstupu: ' + vstup
        tfidf, nazvySoub = pickle.load(open('PomocneSoubory/' + hlSb, "rb"))

    return tfidf, nazvySoub


class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield LabeledSentence(doc, self.labels_list[idx])

def VytvorReprDoc2Vec(vstup, documents, labelss, velikost, okno, alphaa, minalphaa, minimalniPocetCetnostiSlov):

    hlSb = vstup + 'doc2vecModel'+ str(velikost)+ str(okno) + str(alphaa) + str(alphaa)+'.model'
    souboryPS, slozkyPS = ZiskejNazvySouboru('PomocneSoubory/', hlSb)

    train = []
    for i in range(len(labelss)):
        train.append(TaggedDocument(documents[labelss[i]], [labelss[i]]))


    #train = LabeledLineSentence(documents,labelss)
    if len(souboryPS) == 0:
        print 'Vytváření doc2vec modelu vstupu: '+ vstup

        #model = Doc2Vec(size=velikost, window=okno, min_count=5, workers=6, alpha=0.025, min_alpha=0.025)
        model = Doc2Vec(dbow_words=0, dm_concat=0, dm_mean=0, hs=0, negative=5, iter=20, sample=0.0, size=velikost, dm=2, window=okno, min_count=minimalniPocetCetnostiSlov, workers=8, alpha=alphaa, min_alpha=minalphaa)
        #it = LabeledLineSentence(dokumentyPrac, labelss)
        model.build_vocab(train)

        '''
        for epoch in range(2):
            model.train(train)#, total_examples = model.corpus_count, epochs = model.iter)
            model.alpha -= 0.001  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no deca
            model.train(train)
        '''
        model.train(train)
        model.save('PomocneSoubory/' + hlSb)

    else:
        print 'Načítání doc2vec modelu vstupu: ' + vstup
        model = Doc2Vec.load('PomocneSoubory/' + hlSb)

    maticeVys = []
    for vec in model.docvecs:
        maticeVys.append(vec)


    return maticeVys


def VytvorReprDoc2VecTrainTest(vstup, documentsTrain, documentsTest, labelssTrain, labelssTest, velikost, okno, alphaa, minalphaa, minimalniPocetCetnostiSlov):

    hlSb = vstup + 'doc2vecModelTT'+ str(velikost)+ str(okno) + str(alphaa) + str(alphaa)+'.model'
    souboryPS, slozkyPS = ZiskejNazvySouboru('PomocneSoubory/', hlSb)

    full = []
    for i in range(len(labelssTrain)):
        full.append(TaggedDocument(documentsTrain[labelssTrain[i]], [labelssTrain[i]]))
    for i in range(len(labelssTest)):
        full.append(TaggedDocument(documentsTest[labelssTest[i]], [labelssTest[i]]))

    train = []
    for i in range(len(labelssTrain)):
        train.append(TaggedDocument(documentsTrain[labelssTrain[i]], [labelssTrain[i]]))

    test = []
    for i in range(len(labelssTest)):
        test.append(TaggedDocument(documentsTest[labelssTest[i]], [labelssTest[i]]))

    print len(full), len(train), len(test)

    #train = LabeledLineSentence(documents,labelss)
    if len(souboryPS) == 0:
        print 'Vytváření doc2vec modelu vstupu: '+ vstup + "TT"

        #model = Doc2Vec(size=velikost, window=okno, min_count=5, workers=6, alpha=0.025, min_alpha=0.025)
        model = Doc2Vec(dbow_words=0, dm_concat=0, dm_mean=0, hs=0, negative=5, iter=20, sample=0.0, size=velikost, dm=2, window=okno, min_count=minimalniPocetCetnostiSlov, workers=8, alpha=alphaa, min_alpha=minalphaa)
        #it = LabeledLineSentence(dokumentyPrac, labelss)
        model.build_vocab(full)


        model.train(full)
        model.save('PomocneSoubory/' + hlSb)

    else:
        print 'Načítání doc2vec modelu vstupu: ' + vstup
        model = Doc2Vec.load('PomocneSoubory/' + hlSb)

    #print full
    maticeVysTr = []
    for i in range(len(labelssTrain)):
        maticeVysTr.append(model.docvecs[labelssTrain[i]])

    maticeVysTest = []
    for i in range(len(labelssTest)):
        maticeVysTest.append(model.docvecs[labelssTest[i]])

    return maticeVysTr, maticeVysTest



def VytvorReprWord2Vec(vstup, documents, labelss, velikost, okno, alphaa, minalphaa, minimalniPocetCetnostiSlov, slovnik):

    hlSb = vstup + 'word2vecModel'+ str(velikost)+ str(okno) + str(alphaa) + str(alphaa)+'.pkl'
    souboryPS, slozkyPS = ZiskejNazvySouboru('PomocneSoubory/', hlSb)

    train = []
    for i in range(len(labelss)):
        train.append(documents[labelss[i]])
    if len(souboryPS) == 0:
        print 'Vytváření word2vec modelu vstupu: '+ vstup

        model = Word2Vec(train, size=velikost, window=5, min_count=minimalniPocetCetnostiSlov, workers=8, alpha=alphaa)

        vocabu = model.vocab
        matice = []
        for doc in train:
            vektor = [0.0] * velikost
            for slovo in doc:
                if slovnik.has_key(slovo):
                    if vocabu.has_key(slovo):
                        reprSlova = model[slovo]
                        for iv in range(len(vektor)):
                            vektor[iv] = vektor[iv] + reprSlova[iv]
            matice.append(np.divide(np.divide(vektor, float(len(doc))) , np.linalg.norm(np.divide(vektor, float(len(doc))))))

        w2v = dict(zip(model.wv.index2word, model.wv.syn0))

        etree_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                              ("extra trees", ExtraTreesClassifier(n_estimators=200))])
        etree_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
                                    ("extra trees", ExtraTreesClassifier(n_estimators=200))])

        pickle.dump([w2v, etree_w2v, etree_w2v_tfidf, matice], open('PomocneSoubory/' + hlSb, "wb"))

    else:
        print 'Načítání word2vec modelu vstupu: ' + vstup
        w2v, etree_w2v, etree_w2v_tfidf, matice = pickle.load(open('PomocneSoubory/' + hlSb, "rb"))

    return matice


def VytvorReprDoc2VecPredemTrenovanou(vstup, documents, labelss, velikost, okno, alphaa, minalphaa, minimalniPocetCetnostiSlov):
    hlSb = 'TrainCZ' + 'doc2vecModel.model'
    souboryPS, slozkyPS = ZiskejNazvySouboru('PomSoubTrainCZ/', hlSb)

    if not len(souboryPS) == 0:
        print 'Načítání doc2vec modelu vstupu natrénované na češtině.'
        model = Doc2Vec.load('PomSoubTrainCZ/' + hlSb)
        print 'Vytváření matice'
        maticeVys = []
        for i in range(len(labelss)):
            maticeVys.append(model.infer_vector(documents[labelss[i]]))
        print len(documents), len(maticeVys)
        model = 0

    else:

        maticeVys = VytvorReprDoc2Vec(vstup, documents, labelss, velikost, okno, alphaa, minalphaa, minimalniPocetCetnostiSlov)


    return maticeVys


def VytvorReprWord2VecPredemTrenovanou(vstup, documents, labelss, velikost, okno, alphaa, minalphaa, minimalniPocetCetnostiSlov, slovnik):
    hlSb = 'TrainCZ' + 'word2vecModel.model'
    souboryPS, slozkyPS = ZiskejNazvySouboru('PomSoubTrainCZ/', hlSb)
    if not len(souboryPS) == 0:
        train = []
        for i in range(len(labelss)):
            train.append(documents[labelss[i]])
        model = Word2Vec.load('PomSoubTrainCZ/' + hlSb)
        vocabu = model.vocab
        matice = []
        for doc in train:
            vektor = [0.0] * velikost
            for slovo in doc:
                if slovnik.has_key(slovo):
                    if vocabu.has_key(slovo):
                        reprSlova = model[slovo]
                        for iv in range(len(vektor)):
                            vektor[iv] = vektor[iv] + reprSlova[iv]
            matice.append(
                np.divide(np.divide(vektor, float(len(doc))), np.linalg.norm(np.divide(vektor, float(len(doc))))))
    else:
        matice = VytvorReprWord2Vec(vstup, documents, labelss, velikost, okno, alphaa, minalphaa, minimalniPocetCetnostiSlov, slovnik)

    return matice

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(lambda: max_idf, [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

def ZkontrolujJestliJeSlozka(file_path):
    directory = os.path.dirname(file_path)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)

def VstupProNeuNLI(souboryText, vstup, jazyk, soubAslozky, jakyVstup):

    hlSb = vstup + jakyVstup + 'VycisNeuNLIVstup.p'
    souboryPS, slozkyPS = ZiskejNazvySouboru('PomocneSoubory/', hlSb)

    if len(souboryPS) == 0:
        print 'Probíhá tokenizace a POS textů.'
        if jazyk == "english":
            tagger = Tagger.load('english-morphium-wsj-140407.tagger')
        else:
            tagger = Tagger.load('czech-morfflex-pdt-161115.tagger')
        original, vycistene, lemma, tags, lemmaaa = {}, {}, {}, {}, {}
        pocet = 0
        print len(souboryText)
        for keyy in souboryText:
            original[keyy] = souboryText[keyy]
            clanek = souboryText[keyy]
            if pocet % 1000 == 0:
                print pocet
            clanek = clanek.replace(u'.', u' . \n')
            clanek = clanek.replace(u',', u' , \n')
            clanek = clanek.replace(u'!', u' ! \n')
            clanek = clanek.replace(u'?', u' ? \n')
            clanek = clanek.replace(u':', u' : ')
            clanek = clanek.replace(u'"', u' " ')
            clanek = clanek.replace(u"'", u" ' ")
            clanek = clanek.replace(u'˝', u' ˝ ')
            clanek = clanek.replace(u';', u' ; ')
            clanek = clanek.replace(u'(', u' ( ')
            clanek = clanek.replace(u')', u' ) ')
            clanek = clanek.replace(u'[', u' [ ')
            clanek = clanek.replace(u']', u' ] ')
            clanek = clanek.replace(u'{', u' { ')
            clanek = clanek.replace(u'}', u' } ')
            clanek = clanek.replace(u'<', u' < ')
            clanek = clanek.replace(u'>', u' > ')
            clanek = clanek.replace(u'*', u' * ')
            clanek = clanek.replace(u'-', u' - ')
            clanek = clanek.replace(u'   ', u' ')
            clanek = clanek.replace(u'  ', u' ').strip()
            vycistene[keyy], lemmaaa[keyy], tags[keyy] = vytvorLemmaTagsTokens(souboryText[keyy], tagger)
            lemma[keyy] = clanek
            pocet += 1
        pickle.dump([original, vycistene, lemma, tags], open('PomocneSoubory/' + hlSb, "wb"))
    else:
        print hlSb
        print 'Načítání vyčištěného textu, lemmat a tagů ze souboru.'
        original, vycistene, lemma, tags = pickle.load(open('PomocneSoubory/' + hlSb, "rb"))

    file_path = u'data/labels' + jakyVstup + u'/'
    ZkontrolujJestliJeSlozka(file_path)
    file_path = u'data/essays' + jakyVstup + u'/'
    ZkontrolujJestliJeSlozka(file_path)
    file_path = u'data/essays' + jakyVstup + u'/all/'
    ZkontrolujJestliJeSlozka(file_path)
    file_path = u'data/essays' + jakyVstup + u'/train/'
    ZkontrolujJestliJeSlozka(file_path)
    file_path = u'data/essays' + jakyVstup + u'/test/'
    ZkontrolujJestliJeSlozka(file_path)

    file_pathLabelsAll = u'data/labels' + jakyVstup + u'/all/'
    ZkontrolujJestliJeSlozka(file_pathLabelsAll)
    csvfileAll = open(file_pathLabelsAll + 'labels.all.csv', 'wb')
    writerAll = csv.writer(csvfileAll, delimiter=',',
                           quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    writerAll.writerow(['test_taker_id', 'speech_prompt', 'essay_prompt', 'L1'])

    file_pathAllOR = u'data/essays' + jakyVstup + u'/all/original/'
    ZkontrolujJestliJeSlozka(file_pathAllOR)

    file_pathAllPOS = u'data/essays' + jakyVstup + u'/all/POS/'
    ZkontrolujJestliJeSlozka(file_pathAllPOS)

    file_pathAllTOK = u'data/essays' + jakyVstup + u'/all/tokenized/'
    ZkontrolujJestliJeSlozka(file_pathAllTOK)
    for keyyPom in original:
        writerAll.writerow([keyyPom, u'X', u'X', soubAslozky[keyyPom].strip().replace(' ', '_')])
        file0 = file(file_pathAllOR + keyyPom + u'.txt', 'w')
        file0.write(codecs.BOM_UTF8)
        file0.write(original[keyyPom].encode('utf8'))
        file0.close()

        file0 = file(file_pathAllPOS + keyyPom + u'.txt', 'w')
        file0.write(codecs.BOM_UTF8)
        file0.write(u' '.join(tags[keyyPom]).encode('utf8'))
        file0.close()

        file0 = file(file_pathAllTOK + keyyPom + u'.txt', 'w')
        file0.write(codecs.BOM_UTF8)
        file0.write(u' '.join(lemma[keyyPom]).encode('utf8'))
        file0.close()

    file_pathTrainOR = u'data/essays' + jakyVstup + u'/train/original/'
    ZkontrolujJestliJeSlozka(file_pathTrainOR)
    file_pathTrainPOS = u'data/essays' + jakyVstup + u'/train/POS/'
    ZkontrolujJestliJeSlozka(file_pathTrainPOS)
    file_pathTrainTOK = u'data/essays' + jakyVstup + u'/train/tokenized/'
    ZkontrolujJestliJeSlozka(file_pathTrainTOK)

    file_pathTestOR = u'data/essays' + jakyVstup + u'/test/original/'
    ZkontrolujJestliJeSlozka(file_pathTestOR)
    file_pathTestPOS = u'data/essays' + jakyVstup + u'/test/POS/'
    ZkontrolujJestliJeSlozka(file_pathTestPOS)
    file_pathTestTOK = u'data/essays' + jakyVstup + u'/test/tokenized/'
    ZkontrolujJestliJeSlozka(file_pathTestTOK)

    # vytvoření labelů
    file_pathLabelsTrain = u'data/labels' + jakyVstup + u'/train/'
    ZkontrolujJestliJeSlozka(file_pathLabelsTrain)
    file_pathLabelsTest = u'data/labels' + jakyVstup + u'/test/'
    ZkontrolujJestliJeSlozka(file_pathLabelsTest)

    csvfileTrain = open(file_pathLabelsTrain + 'labels.train.csv', 'wb')
    writerTrain = csv.writer(csvfileTrain, delimiter=',',
                                quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    writerTrain.writerow(['test_taker_id', 'speech_prompt', 'essay_prompt', 'L1'])

    csvfileTest = open(file_pathLabelsTest + 'labels.test.csv', 'wb')
    writerTest = csv.writer(csvfileTest, delimiter=',',
                                quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    writerTest.writerow(['test_taker_id', 'speech_prompt', 'essay_prompt', 'L1'])

    pocetSouboru = len(vycistene)
    trainPoc = int((pocetSouboru / 100.0) * 80.0)
    pocet = 0
    labely = {}
    for keyyPom in original:
        if not labely.has_key(soubAslozky[keyyPom].strip().replace(' ', '_')):
            labely[soubAslozky[keyyPom].strip().replace(' ', '_')] = soubAslozky[keyyPom].strip().replace(' ', '_')
        if pocet < trainPoc:
            writerTrain.writerow([keyyPom, u'X', u'X', soubAslozky[keyyPom].strip().replace(' ', '_')])
            file0 = file(file_pathTrainOR + keyyPom + u'.txt', 'w')
            file0.write(codecs.BOM_UTF8)
            file0.write(original[keyyPom].encode('utf8'))
            file0.close()

            file0 = file(file_pathTrainPOS + keyyPom + u'.txt', 'w')
            file0.write(codecs.BOM_UTF8)
            file0.write(u' '.join(tags[keyyPom]).encode('utf8'))
            file0.close()

            file0 = file(file_pathTrainTOK + keyyPom + u'.txt', 'w')
            file0.write(codecs.BOM_UTF8)
            file0.write(u' '.join(lemma[keyyPom]).encode('utf8'))
            file0.close()
        else:
            writerTest.writerow([keyyPom, u'X', u'X', soubAslozky[keyyPom].strip().replace(' ', '_')])
            file0 = file(file_pathTestOR + keyyPom + u'.txt', 'w')
            file0.write(codecs.BOM_UTF8)
            file0.write(original[keyyPom].encode('utf8'))
            file0.close()

            file0 = file(file_pathTestPOS + keyyPom + u'.txt', 'w')
            file0.write(codecs.BOM_UTF8)
            file0.write(u' '.join(tags[keyyPom]).encode('utf8'))
            file0.close()

            file0 = file(file_pathTestTOK + keyyPom + u'.txt', 'w')
            file0.write(codecs.BOM_UTF8)
            file0.write(u' '.join(lemma[keyyPom]).encode('utf8'))
            file0.close()

        pocet += 1

    file0 = file(u'data/' + vstup + jakyVstup + u'.txt', 'w')
    file0.write(codecs.BOM_UTF8)
    for lab in labely:
        file0.write(u"'".encode('utf8'))
        file0.write(unicode(lab.decode('utf8')).encode('utf8'))
        file0.write(u"', ".encode('utf8'))
    file0.close()

    csvfileAll.close()
    csvfileTrain.close()
    csvfileTest.close()

    print 'Vytvoření vstupů pro NLI neuronovky dokončeno. '