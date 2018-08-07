
from sklearn import svm

pocSoub = len(soubAslozky)
pocCrossValid = 10
pocTrain = int((pocSoub / 100.0) * 75.0)
pocTest = pocSoub - pocTrain
skokPozZacTrain = int(pocSoub / float(pocCrossValid))

tfidfMat = tfidfMat.toarray()

for crossValid in range(pocCrossValid):
    pocPozTr = crossValid * skokPozZacTrain
    pocPozTs = pocPozTr+pocTrain
    print 'Cyklus: ' + str(crossValid)
    if pocPozTs <= pocSoub:
        soubTrain = nazvySoub[pocPozTr:pocPozTs]
        pozadVystupTrain = pozadVystup[pocPozTr:pocPozTs]
        tfidfTrain = tfidfMat[pocPozTr:pocPozTs]
    else:
        konTr = pocTrain - len(nazvySoub[pocPozTr:pocSoub-1])
        soubTrain = nazvySoub[pocPozTr:pocSoub-1] + nazvySoub[0:konTr]
        pozadVystupTrain = pozadVystup[pocPozTr:pocSoub-1] + pozadVystup[0:konTr]
        tfidfTrain = np.append(tfidfMat[pocPozTr:pocSoub-1], tfidfMat[0:konTr], axis=0)

    if pocPozTs + pocTest <= pocSoub:
        soubTest = nazvySoub[pocPozTs:pocPozTs + pocTest]
        tfidfTest = tfidfMat[pocPozTs:pocPozTs + pocTest]
    else:
        konTs = pocTest - len(nazvySoub[pocPozTs:pocSoub - 1])
        soubTest = nazvySoub[pocPozTs:pocSoub - 1] + nazvySoub[0:konTs]
        tfidfTest = np.append(tfidfMat[pocPozTs:pocSoub - 1], tfidfMat[0:konTs], axis=0)

    soubAslozkyTest = {}
    for soub in soubTest:
        soubAslozkyTest[soub] = soubAslozky[soub]
    # tfidf
    clf = svm.LinearSVC()
    clf.fit(tfidfTrain, pozadVystupTrain)
    vysledekTFIDF = clf.predict(tfidfTest)