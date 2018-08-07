# -*- coding: utf-8 -*-
# coding: utf-8
import random, math, time, numpy
from scipy.spatial import distance

def magnitudeGWM(v):
    return math.sqrt(sum(v[i]*v[i] for i in range(len(v))))

def normalizeGWM(v):
    vmag = magnitudeGWM(v)
    return [ v[i]/vmag  for i in range(len(v)) ]


#The Grow When Required network
def natrenujNeuGWM(vstup, pocetShluku):
    A = []
    h = []
    for i in range(2):
        poz = random.randint(0,len(vstup)-1)
        A.append(vstup[poz])
        h.append(1.0)
    C = {}
    age = {}
    aT = 1.5
    hT = 0.5
    alfb = 1.05
    alfn = 1.05
    taub = 3.33
    taun = 14.3
    ageMax = 10.0
    h0 = 1.0
    iterace = 1
    konec = 0
    while konec == 0:
        print iterace, len(A)
        if iterace%100 == 0:
            print iterace, len(A)
        for x in vstup:
            #step 2
            vzdalenostiNodu = []
            poziceNodu = []
            for i in range(len(A)):
                w = A[i]
                vzdalenostiNodu.append(distance.euclidean(x, w))
                poziceNodu.append(i)
            #step 3
            for i in range(len(vzdalenostiNodu)):
                if i+1 < len(vzdalenostiNodu):
                    for j in range(i+1,len(vzdalenostiNodu)):
                        if vzdalenostiNodu[i] > vzdalenostiNodu[j]:
                            pom = vzdalenostiNodu[i]
                            pom2 = poziceNodu[i]
                            vzdalenostiNodu[i] = vzdalenostiNodu[j]
                            poziceNodu[i] = poziceNodu[j]
                            vzdalenostiNodu[j] = pom
                            poziceNodu[j] = pom2

            #step 4
            s = poziceNodu[0]
            t = poziceNodu[1]
            if not C.has_key((s, t)):
                C[(s, t)] = 1.0
            else:
                C[(s, t)] = 0.0

            #step 5
            act = numpy.exp(-vzdalenostiNodu[0])

            if act < aT and act < hT: #and len(A) < pocetShluku:
                #step 6
                wr = (A[s] + x) / 2.0
                A.append(wr)
                h.append(1.0)
                C[(len(A)-1, s)] = 1.0
                C[(len(A)-1, t)] = 1.0
                C.pop((s, t))
            else:
                #step 7
                A[s] = 0.6 * h[s] * (x - A[s])
                if not s == 0 and not s == len(A)-1:
                    A[s-1] = 0.4 * h[s-1] * (x-A[s-1])
                    A[s+1] = 0.4 * h[s-1] * (x - A[s+1])
                elif s == 0:
                    A[len(A) - 1] = 0.4 * h[len(A) - 1] * (x - A[len(A) - 1])
                    A[s + 1] = 0.4 * h[s - 1] * (x - A[s + 1])
                elif s == len(A)-1:
                    A[s - 1] = 0.4 * h[s - 1] * (x - A[s - 1])
                    A[0] = 0.4 * h[0] * (x - A[0])


            #step 8
            if age.has_key((s, s - 1)):
                age[(s, s - 1)] = age[(s, s - 1)] + 1
            else:
                age[(s, s - 1)] = 1
            if age.has_key((s, s + 1)):
                age[(s, s + 1)] = age[(s, s + 1)] + 1
            else:
                age[(s, s + 1)] = 1

            #step 9
            h[s] = h0 - (1.0 / alfb) * (1.0 - math.pow(math.e, alfb / taub))
            if not s == 0 and not s == len(h) - 1:
                h[s - 1] = h0 - (1.0 / alfb) * (1.0 - math.pow(math.e, alfb / taun))
                h[s + 1] = h0 - (1.0 / alfn) * (1.0 - math.pow(math.e, alfn / taun))
            elif s == 0:
                h[len(h)-1] = h0 - (1.0 / alfb) * (1.0 - math.pow(math.e, alfb / taun))
                h[s + 1] = h0 - (1.0 / alfn) * (1.0 - math.pow(math.e, alfn / taun))
            elif s == len(h)-1:
                h[s - 1] = h0 - (1.0 / alfb) * (1.0 - math.pow(math.e, alfb / taun))
                h[0] = h0 - (1.0 / alfn) * (1.0 - math.pow(math.e, alfn / taun))

            #step 10
            soused = {}
            for cc in C:
                for c in cc:
                    if not soused.has_key(c):
                        soused[c] = c
            coPop = {}
            for i in range(len(A)):
                if not soused.has_key(i):
                    if not coPop.has_key(i):
                        coPop[i] = i
            for ag in age:
                if age[ag] > ageMax:
                    if not coPop.has_key(ag[0]):
                        coPop[ag[0]] = ag[0]
                    if not coPop.has_key(ag[1]):
                        coPop[ag[1]] = ag[1]
                    if C.has_key(ag):
                        C.pop(ag)
            if len(A) > 2:
                if len(A) - len(coPop) > 2:
                    Apom = []
                    hPom = []
                    for i in range(len(A)):
                        if not coPop.has_key(i):
                            Apom.append(A[i])
                            hPom.append(h[i])
                    A = Apom
                    h = hPom
        iterace += 1
        if iterace > 1000 and len(A) > 19 and len(A) < 25:
            konec = 1

    print len(A)
    centroids = []
    for a in A:
        centroids.append(a)
    return centroids

def klasifikujNeuGWR(vstup, centroids):
    vysledek = []
    for x in vstup:
        minVz = 100000000000.0
        minVzCen = 0
        for j in range(len(centroids)):
            w = centroids[j]
            dst = distance.euclidean(x, w)
            if dst < minVz:
                minVz = dst
                minVzCen = j

        vysledek.append(minVzCen)
    return vysledek
