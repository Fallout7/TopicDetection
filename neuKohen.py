# -*- coding: utf-8 -*-
# coding: utf-8
import random, math, time
from scipy.spatial import distance

def magnitude(v):
    return math.sqrt(sum(v[i]*v[i] for i in range(len(v))))

def normalize(v):
    vmag = magnitude(v)
    return [ v[i]/vmag  for i in range(len(v)) ]



def natrenujNeu(vstup, pocetShluku):
    centroids = []
    pocPriz = len(vstup[0])
    for i in range(5000):
        cPom = []
        for j in range(pocPriz):
            cPom.append(random.uniform(0.0,10.0))
        centroids.append(normalize(cPom))
    centroids2 = []
    pocPriz = len(vstup[0])
    for i in range(pocetShluku):
        cPom = []
        for j in range(pocPriz):
            cPom.append(random.uniform(0.0, 10.0))
        centroids2.append(normalize(cPom))

    cc = 0.01
    pocetBehu = 10000
    for i in range(pocetBehu):
        if i%1000 == 0:
            print i
        for x in vstup:
            minVz = 100000000000.0
            minVzCen = 0
            for j in range(len(centroids)):
                w = centroids[j]
                dst = distance.euclidean(x, w)
                if dst < minVz:
                    minVz = dst
                    minVzCen = j

        centroids[minVzCen] = normalize(centroids[minVzCen] + cc * (x - centroids[minVzCen]))

        for x in vstup:
            minVz = 100000000000.0
            minVzCen = 0
            for j in range(len(centroids)):
                w = centroids[j]
                dst = distance.euclidean(x, w)
                if dst < minVz:
                    minVz = dst
                    minVzCen = j

        centroids[minVzCen] = normalize(centroids[minVzCen] + cc * (x - centroids[minVzCen]))

    return centroids

def klasifikujNeu(vstup, centroids):
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
