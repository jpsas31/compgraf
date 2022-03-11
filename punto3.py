#Implementar el algoritmo del agente
# viajero para 10 ciudades y hallar el mínimo camino del
# agente para recorrer toda la zona de ciudades.
# Distancia mínima entre ciudades 1km
# Distancia máxima entre ciudades 20km.
# https://es.wikipedia.org/wiki/Problema_del_viajante
# Para este algoritmo se debe de realizar una matriz de 10x10,
# donde se relacione la distancia entre ciudades, esta distancia
# se puede colocar al aleatoriamente, la primera vez que se
# ejecute el algoritmo pero se debe usar la misma matriz para
# todas las repeticiones, para que la mínima distancia a hallar
# (minimización de la distancia) sea siempre la misma.
# El algoritmo debe arrojar, la distancia total mínima
# recorrido por el agente viajero, y además el orden de las
# ciudades a visitar para lograr el mínimo de distancia.

import random
import numpy 
import pandas as pd
import itertools


ciudades=['Bogotá','Medellín', 'Cali', 'Barranquilla', 'Cartagena ', 'Cúcuta ', 'Soacha ', 'Soledad ', 'Bucaramanga', 'Bello']

def grafoLabels(grafo,ciudades):
    dicta={}
    for j,col in enumerate(grafo):
        dicta[ciudades[j]]=col

    a=pd.DataFrame.from_dict(dicta,orient='index', columns=ciudades)
    print(a)

def matrizDistancias(ciudades):
    matriz= numpy.zeros([len(ciudades),len(ciudades)])
    for ciudad in range(len(ciudades)):
        for ciudad2 in range(len(ciudades)):
            if(ciudad != ciudad2):
                matriz[ciudad][ciudad2]=random.randrange(1,20)
            else:
                matriz[ciudad][ciudad2]=0
    return matriz   

def getdistPath(path,dist,ciudades):
    suma=0
    for c in range(len(path)-1):
        suma+=dist[ciudades.index(path[c])][ciudades.index(path[c+1])]
    return suma

def subsets(size,lista):
    return itertools.combinations(lista, size)

#encuentra la distancia minima de k  a una ciudad
def minDist(s,k,c,dist,ciudades):
    minval=100000
    s=list(s)
    s.pop(s.index(k))
    pos=-1
    for m in s:
        if(minval > c[" ".join(s) + str(ciudades.index(m))][0]+dist[ciudades.index(m)][ciudades.index(k)]):
            minval= c[" ".join(s) + str(ciudades.index(m))][0]+dist[ciudades.index(m)][ciudades.index(k)]
            pos=" ".join(s) + str(ciudades.index(m))

    return minval,pos
    
#Retorna el ciclo mas corto del set
def shortestPath(c,dist,ciudades):
    val = minDist(ciudades,ciudades[0],c,dist,ciudades)
    path=getpath(c,val[1],ciudades)
    return [val[0],path]

#Contruye el camino 
def getpath(c,initial,ciudades):
    path=[]
    path.append(ciudades[int(initial[-1])])
    x=initial
    while True:
        x=c[x][1]
        if(x==0):break
        path.append(ciudades[int(x[-1])])
    path.reverse()
    path.insert(0,ciudades[0])
    path.append(ciudades[0])
    return path


##Version dinamica del algoritmo
def TSP (ciudades,grafo):
    
    n= grafo.shape[0]
    C={}

    for k in range(1,n,1):
        C[ciudades[k]+ str(k)] = [grafo[0][k],0]
   
    for i in range(2,n,1):
        for subset in subsets(i,ciudades[1:]):
            for k in subset:
                val=minDist(subset,k,C,grafo,ciudades)
                C[" ".join(list(subset)) + str(ciudades.index(k))]= val
   
    return shortestPath(C,grafo,ciudades)

##Version de fuerza bruta del algoritmo
##Revisa todas las posibles combinaciones
def bruteforce(ciudades,dist):
    minval=1000
    minpath=[]
    for i in itertools.permutations(ciudades[1:],len(ciudades)-1):
        i=list(i)
        i.insert(0,ciudades[0])
        i.append(ciudades[0])
        val=getdistPath(i,dist,ciudades)
        if(val<minval):
            minval=val
            minpath=i
    
    return [minval,minpath]


grafo=matrizDistancias(ciudades)
grafoLabels(grafo,ciudades)
print("bruto",bruteforce(ciudades,grafo))
print("dinamico",TSP(ciudades,grafo))