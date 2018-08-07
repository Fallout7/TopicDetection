# -*- coding: utf-8 -*-
# coding: utf-8

from predpriprava import *
from sklearn import svm
from evaluation import *
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

def ExtrakceCastiMatice(nazvySoub, soubAslozky, pocTrainShl, tfidfMat, pozadVystup, skokPozZacTrainShl, crossValid, propojeni):
    tfidfShl = {}
    pozadVystupShl = {}
    souboryShl = {}
    pocVshlukuSouboru = {}
    shlJe = {}
    for ii in range(len(nazvySoub)):
        soubb = nazvySoub[ii]
        origshl = soubAslozky[soubb]
        if not shlJe.has_key(origshl):
            shlJe[origshl] = origshl
            tfidfShl[origshl] = []
            pozadVystupShl[origshl] = []
            souboryShl[origshl] = []
            tfidfShl[origshl] = tfidfShl[origshl] + [tfidfMat[ii]]
            pozadVystupShl[origshl] = pozadVystupShl[origshl] + [pozadVystup[ii]]
            souboryShl[origshl] = souboryShl[origshl] + [nazvySoub[ii]]
            pocVshlukuSouboru[origshl] = 0
        else:
            tfidfShl[origshl] = tfidfShl[origshl] + [tfidfMat[ii]]
            pozadVystupShl[origshl] = pozadVystupShl[origshl] + [pozadVystup[ii]]
            souboryShl[origshl] = souboryShl[origshl] + [nazvySoub[ii]]
            pocVshlukuSouboru[origshl] = pocVshlukuSouboru[origshl] + 1

    tfidfTrain = []
    tfidfTest = []
    pozadVystupTrain = []
    pozadVystupTest = []
    soubTest = []
    souborUzjeTest = {}

    pocTrain = 0
    for origShl in pocTrainShl:
        pocTrain += pocTrainShl[origShl]
        vysShl = propojeni[origShl]
        pocPozTr = int(skokPozZacTrainShl[origShl] * crossValid)
        #pocPozTs = pocPozTr + pocTrainShl[origShl]
        pocetTestDat = pocVshlukuSouboru[origShl] - pocTrainShl[origShl]

        matShl = tfidfShl[origShl]
        soubbShl = souboryShl[origShl]
        pozadovanyVystupShl = pozadVystupShl[origShl]
        pocTrainPriz = 0
        pocTestPriz = 0
        for ii in range(len(matShl)):
            if ii >= pocPozTr:
                if pocTrainPriz < pocTrainShl[origShl]:
                    tfidfTrain.append(matShl[ii])
                    pozadVystupTrain.append(pozadovanyVystupShl[ii])
                    pocTrainPriz += 1
                else:
                    if pocTestPriz <= pocetTestDat:
                        '''
                        if not souborUzjeTest.has_key(soubbShl[ii]):
                            souborUzjeTest[soubbShl[ii]] = soubbShl[ii]
                        '''
                        tfidfTest.append(matShl[ii])
                        pozadVystupTest.append(pozadovanyVystupShl[ii])
                        soubTest.append(soubbShl[ii])
                        pocTestPriz += 1
        for ii in range(len(matShl)):
            if pocTrainPriz < pocTrainShl[origShl]:
                tfidfTrain.append(matShl[ii])
                pozadVystupTrain.append(pozadovanyVystupShl[ii])
                pocTrainPriz += 1
            else:
                if pocTestPriz <= pocetTestDat:
                    '''
                    if not souborUzjeTest.has_key(soubbShl[ii]):
                        souborUzjeTest[soubbShl[ii]] = soubbShl[ii]
                    '''
                    tfidfTest.append(matShl[ii])
                    pozadVystupTest.append(pozadovanyVystupShl[ii])
                    soubTest.append(soubbShl[ii])
                    pocTestPriz += 1

    pocShluku = 0
    uzJeSl = {}
    for keyy in soubAslozky:
        if not uzJeSl.has_key(soubAslozky[keyy]):
            uzJeSl[soubAslozky[keyy]] = soubAslozky[keyy]
            pocShluku += 1

    # provedení shuffle train dat
    tfidfTrainSh = []
    pozadVystupTrainSh = []
    pouzitePoz = {}
    kon = 0
    while kon == 0:
        pozz = random.randint(0, len(tfidfTrain) - 1)
        if not pouzitePoz.has_key(pozz):
            pouzitePoz[pozz] = pozz
            tfidfTrainSh.append(tfidfTrain[pozz])
            pozadVystupTrainSh.append(pozadVystupTrain[pozz])
        if len(tfidfTrainSh) == len(tfidfTrain):
            kon = 1

    tfidfTrain = tfidfTrainSh
    pozadVystupTrain = pozadVystupTrainSh


    soubAslozkyTest = {}
    for soub in soubTest:
        soubAslozkyTest[soub] = soubAslozky[soub]

    train_matrix_essays = np.array(tfidfTrain)
    test_matrix_essays = np.array(tfidfTest)

    return train_matrix_essays, test_matrix_essays, pozadVystupTrain, soubAslozkyTest, pozadVystupTest, pocShluku, soubTest

def ExtrakceCastiMaticeUnsup(nazvySoub, soubAslozky, pocTrainShl, tfidfMat, pozadVystup, skokPozZacTrainShl, crossValid, propojeni, documents):
    tfidfShl = {}
    pozadVystupShl = {}
    souboryShl = {}
    pocVshlukuSouboru = {}
    shlJe = {}
    for ii in range(len(nazvySoub)):
        soubb = nazvySoub[ii]
        origshl = soubAslozky[soubb]
        if not shlJe.has_key(origshl):
            shlJe[origshl] = origshl
            tfidfShl[origshl] = []
            pozadVystupShl[origshl] = []
            souboryShl[origshl] = []
            tfidfShl[origshl] = tfidfShl[origshl] + [tfidfMat[ii]]
            pozadVystupShl[origshl] = pozadVystupShl[origshl] + [pozadVystup[ii]]
            souboryShl[origshl] = souboryShl[origshl] + [nazvySoub[ii]]
            pocVshlukuSouboru[origshl] = 0
        else:
            tfidfShl[origshl] = tfidfShl[origshl] + [tfidfMat[ii]]
            pozadVystupShl[origshl] = pozadVystupShl[origshl] + [pozadVystup[ii]]
            souboryShl[origshl] = souboryShl[origshl] + [nazvySoub[ii]]
            pocVshlukuSouboru[origshl] = pocVshlukuSouboru[origshl] + 1

    tfidfTrain = []
    tfidfTest = []
    pozadVystupTrain = []
    pozadVystupTest = []
    soubTest = []
    textyPracovniTrain = []
    textyPracovniTest = []

    pocTrain = 0
    for origShl in pocTrainShl:
        pocTrain += pocTrainShl[origShl]
        vysShl = propojeni[origShl]
        pocPozTr = int(skokPozZacTrainShl[origShl] * crossValid)
        #pocPozTs = pocPozTr + pocTrainShl[origShl]
        pocetTestDat = pocVshlukuSouboru[origShl] - pocTrainShl[origShl]

        matShl = tfidfShl[origShl]
        soubbShl = souboryShl[origShl]
        pozadovanyVystupShl = pozadVystupShl[origShl]
        pocTrainPriz = 0
        pocTestPriz = 0
        for ii in range(len(matShl)):
            if ii >= pocPozTr:
                if pocTrainPriz < pocTrainShl[origShl]:
                    tfidfTrain.append(matShl[ii])
                    textyPracovniTrain.append(documents[ii])
                    pozadVystupTrain.append(pozadovanyVystupShl[ii])
                    pocTrainPriz += 1
                else:
                    if pocTestPriz <= pocetTestDat:
                        tfidfTest.append(matShl[ii])
                        textyPracovniTest.append(documents[ii])
                        pozadVystupTest.append(pozadovanyVystupShl[ii])
                        soubTest.append(soubbShl[ii])
                        pocTestPriz += 1
        for ii in range(len(matShl)):
            if pocTrainPriz < pocTrainShl[origShl]:
                tfidfTrain.append(matShl[ii])
                textyPracovniTrain.append(documents[ii])
                pozadVystupTrain.append(pozadovanyVystupShl[ii])
                pocTrainPriz += 1
            else:
                if pocTestPriz <= pocetTestDat:
                    tfidfTest.append(matShl[ii])
                    textyPracovniTest.append(documents[ii])
                    pozadVystupTest.append(pozadovanyVystupShl[ii])
                    soubTest.append(soubbShl[ii])
                    pocTestPriz += 1

    pocShluku = 0
    uzJeSl = {}
    for keyy in soubAslozky:
        if not uzJeSl.has_key(soubAslozky[keyy]):
            uzJeSl[soubAslozky[keyy]] = soubAslozky[keyy]
            pocShluku += 1

    # provedení shuffle train dat
    tfidfTrainSh = []
    textyPracovniTrainSh = []
    pozadVystupTrainSh = []
    pouzitePoz = {}
    kon = 0
    while kon == 0:
        pozz = random.randint(0, len(tfidfTrain) - 1)
        if not pouzitePoz.has_key(pozz):
            pouzitePoz[pozz] = pozz
            tfidfTrainSh.append(tfidfTrain[pozz])
            textyPracovniTrainSh.append(textyPracovniTrain[pozz])
            pozadVystupTrainSh.append(pozadVystupTrain[pozz])
        if len(tfidfTrainSh) == len(tfidfTrain):
            kon = 1

    tfidfTrain = tfidfTrainSh
    textyPracovniTrain = textyPracovniTrainSh
    pozadVystupTrain = pozadVystupTrainSh


    soubAslozkyTest = {}
    for soub in soubTest:
        soubAslozkyTest[soub] = soubAslozky[soub]

    train_matrix_essays = np.array(tfidfTrain)
    test_matrix_essays = np.array(tfidfTest)

    return train_matrix_essays, test_matrix_essays, pozadVystupTrain, soubAslozkyTest, pozadVystupTest, pocShluku, soubTest, textyPracovniTrain, textyPracovniTest

def svmCast(tfidfTrain, pozadVystupTrain, tfidfTest, soubAslozkyTest, soubTest):
    clf = svm.LinearSVC()
    clf.fit(tfidfTrain, pozadVystupTrain)
    vysledekTFIDF = clf.predict(tfidfTest)
    pravdep = clf.decision_function(tfidfTest)
    print vysledekTFIDF[0]
    print pravdep[0]
    time.sleep(10)
    acc = evaluateResultsAcc2(soubAslozkyTest, vysledekTFIDF, soubTest)
    return acc


def neuProved(tfidfTrain, tfidfTest, pozadVystupTrain, pozadVystupTest, num_classes, train_partition_name, epochs, batch_size, soubAslozkyTest, soubTest, dropout, dropout2, un, un2):
    train_matrix_essays = tfidfTrain
    test_matrix_essays = tfidfTest
    # -------------------------------------------------------------------------------------
    # -----------------NN classifier... on essays--------------------------------------------
    # -------------------------------------------------------------------------------------

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(pozadVystupTrain, num_classes)
    y_test = keras.utils.to_categorical(pozadVystupTest, num_classes)

    model_path = ('PomocneSoubNeu/NNessey_model1_trigram50tis_{ep}epochs{ba}batch.h5')
    train_model_path = model_path.format(partition=train_partition_name, ep=epochs, ba=batch_size)

    x_train = train_matrix_essays
    x_test = test_matrix_essays

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    model1 = Sequential()  # uz neni jina moznost modelu
    model1.add(Dense(units=un, activation='tanh', input_dim=x_train.shape[1], name='dense_1'))
    model1.add(Dropout(dropout, name='dropout_1'))
    #model1.add(Dense(units=un2, activation='softsign', name='dense_2'))
    #model1.add(Dropout(dropout2, name='dropout_2'))
    model1.add(Dense(num_classes, activation='softmax', name='dense_5'))

    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model1.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd,
                   metrics=['accuracy'])  # nstaveni ucici algoritmus

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=10, verbose=0, mode='auto',
                                                  cooldown=0, min_lr=0.001)

    history1 = model1.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(x_test,
                                           y_test), callbacks=[
            reduce_lr])  # natrenuj  .. v priade nevejde do mpameti ...  misto fit train_on_batch (nutne zabespecit nastaveni trenovani)

    score1 = model1.evaluate(x_test, y_test, verbose=0)  # vypocitej    print('Test loss:', score1[0])
    '''
    # list all data in history
    # print(history1.history.keys())
    # summarize history for accuracy
    plt.plot(history1.history['acc'])
    plt.plot(history1.history['val_acc'])
    plt.title('model3 accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.figure()
    plt.plot(history1.history['loss'])
    plt.plot(history1.history['val_loss'])
    plt.title('model1 loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    '''
    predicted_probs = model1.predict(x_test)
    predicted = np.argmax(predicted_probs, axis=1)

    accNeu = evaluateResultsAcc2(soubAslozkyTest, predicted, soubTest)
    print('Výsledek NEU: ' + str(accNeu))

    return score1[1], accNeu


