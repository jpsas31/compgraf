# PUNTO 2. (20%)Algoritmo difusión enfermedades
# Modelos de difusión de enfermedades.
# Ver diapositiva.
# https://es.wikipedia.org/wiki/Modelaje_matem%C3%A1tico_de_epidemias
# Realizar un algoritmo en python que acepte un grafo
# cualquiera en formato pajek (LIBRERIA NETWORKX),
# este algoritmo debe Tener:
# entradas.
# Definir la cantidad de nodos semillas
# Definir probabilidad de contagio por vecino.
# Definir pasos de contagios para poder comparar.
# Definir el número de iteraciones por prueba.
# Las semillas se escogen aleatoriamente antes de difundir la
# enfermedad
# El algoritmo debe identificar cuales son los mejores nodos
# de contagio después de n iteraciones.
# Para generar un contagio con difusión en cascada
# independiente se debe medir el contagio después de dos o
# tres pasos de contagio (o sea se sale de la semilla, y se
# contagia y luego los nuevos contagiados contagian a otros y
# se cuenta cuantos van.

from turtle import color
import networkx
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def grafoDistancias(ciudades):
    distancias= networkx.Graph()
    for ciudad in ciudades:
        for ciudad2 in ciudades:
            if(ciudad != ciudad2):
             distancias.add_edge(ciudad, ciudad2, weight=random.randrange(1,20))

    return distancias
def readGraf(filename):
    return networkx.read_pajek(filename)

def dibujarGrafo(g):
    pos = networkx.spring_layout(g, seed=7)
    # print(pos)
    
    networkx.draw_networkx(g,pos)
    # networkx.draw_networkx_nodes(g,pos, node_size=700)
    # networkx.draw_networkx_edges(g, pos, edgelist=g.edges(data=True), width=6)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def dibujarGrafoInfeccion(g,i):
    plt.ion()
    pos = networkx.spring_layout(g, seed=7)
    estado={}
    for persona in pos.items():
        estado[persona[0]]=0
    
  
    ax = plt.gca()
    
    for dia in range(0,len(i),1):
        if(dia==0):infectadosDia=i[dia]
        else: infectadosDia= math.floor(i[dia] - i[dia-1])
        
        if(infectadosDia>0):
            for nodo in estado.keys():
                if(infectadosDia >= 0 and estado[nodo]==0):
                    estado[nodo]=1
                    infectadosDia=infectadosDia-1

        
        rojo=(1,0,0)
        azul=(0,0,1)
        colors=[]
        for nodo in estado.values():
            if(nodo==0): colors.append(azul)
            else: colors.append(rojo)
        
        plt.clf()
        networkx.draw(g,pos,node_color=colors)
        ax.figure.canvas.draw()
        ax.figure.canvas.flush_events()
        plt.pause(0.05)
        plt.show()



        

def modeloSi(graf):
    # Total population, N.
    N = len(graf.nodes)
    # print(N)
    # Initial number of infected and recovered individuals, I0 
    I0 = 200
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 
    # Contact rate, beta.
    beta = 10
    # A grid of time points (in days)
    t = np.linspace(0, 160, 160)

    # The SIR model differential equations.
    def deriv(y, t, N, beta):
        S, I = y    
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N 
    
        return dSdt, dIdt

    # Initial conditions vector
    y0 = S0, I0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta))
    
    S, I = ret.T
    print(I)

    dibujarGrafoInfeccion(graf,I)
    # Plot the data on three separate curves for S(t), I(t) and R(t)
    # fig = plt.figure(facecolor='w')
    # ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    # ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
    # ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
    # ax.set_xlabel('Time /days')
    # ax.set_ylabel('Number (1000s)')
    # ax.set_ylim(0,1.2)
    # ax.yaxis.set_tick_params(length=0)
    # ax.xaxis.set_tick_params(length=0)
    # ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    # legend = ax.legend()
    # legend.get_frame().set_alpha(0.5)
    # for spine in ('top', 'right', 'bottom', 'left'):
    #     ax.spines[spine].set_visible(False)
    # plt.show()


# modeloSi(readGraf('./datasets prueba/USAir97.net'))
modeloSi(readGraf('./datasets prueba/uv.net'))

