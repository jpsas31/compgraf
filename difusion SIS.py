#!/usr/bin/env python
import networkx as nx
import math
import numpy
import csv
import random as rand
import random
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.pyplot as blt
import copy
import matplotlib.animation as animation

print (time.strftime("%I:%M:%S"))

# Animation funciton
def animate(i):
    colors = ['r', 'b', 'g', 'y', 'w', 'm']
    nx.draw_circular(G, node_color=[random.choice(colors) for j in range(9)])




def independent_cascade(G, seeds, steps=0):
  """Return the active nodes of each diffusion step by the independent cascade
  model

  Parameters
  -----------
  G : graph
    A NetworkX graph
  seeds : list of nodes
    The seed nodes for diffusion
  steps: integer
    The number of steps to diffuse.  If steps <= 0, the diffusion runs until
    no more nodes can be activated.  If steps > 0, the diffusion runs for at
    most "steps" rounds

  Returns
  -------
  layer_i_nodes : list of list of activated nodes
    layer_i_nodes[0]: the seeds
    layer_i_nodes[k]: the nodes activated at the kth diffusion step

  Notes
  -----
  When node v in G becomes active, it has a *single* chance of activating
  each currently inactive neighbor w with probability p_{vw}

  Examples
  --------
  >>> DG = nx.DiGraph()
  >>> DG.add_edges_from([(1,2), (1,3), (1,5), (2,1), (3,2), (4,2), (4,3), \
  >>>   (4,6), (5,3), (5,4), (5,6), (6,4), (6,5)], act_prob=0.2)
  >>> H = nx.independent_cascade(DG,[6])

  References
  ----------
  [1] David Kempe, Jon Kleinberg, and Eva Tardos.
      Influential nodes in a diffusion model for social networks.
      In Automata, Languages and Programming, 2005.
  """
  if type(G) == nx.MultiGraph or type(G) == nx.MultiDiGraph:
      raise Exception( \
          "independent_cascade() is not defined for graphs with multiedges.")

  # make sure the seeds are in the graph
  for s in seeds:
    if s not in G.nodes():
      raise Exception("seed", s, "is not in graph")

  # change to directed graph
  if not G.is_directed():
    DG = G.to_directed()
  else:
    DG = copy.deepcopy(G)

  # init activation probabilities
  for e in DG.edges():
    if 'act_prob' not in DG[e[0]][e[1]]:
      DG[e[0]][e[1]]['act_prob'] = 1
    elif DG[e[0]][e[1]]['act_prob'] > 1:
      raise Exception("edge activation probability:", \
          DG[e[0]][e[1]]['act_prob'], "cannot be larger than 1")

  # perform diffusion
  A = copy.deepcopy(seeds)  # prevent side effect
  if steps <= 0:
    # perform diffusion until no more nodes can be activated
    return _diffuse_all(DG, A)
  # perform diffusion for at most "steps" rounds
  return _diffuse_k_rounds(DG, A, steps)

def _diffuse_all(G, A):
  tried_edges = set()
  layer_i_nodes = [ ]
  layer_i_nodes.append([i for i in A])  # prevent side effect
  while True:
    len_old = len(A)
    (A, activated_nodes_of_this_round, cur_tried_edges) = \
        _diffuse_one_round(G, A, tried_edges)
    layer_i_nodes.append(activated_nodes_of_this_round)
    tried_edges = tried_edges.union(cur_tried_edges)
    if len(A) == len_old:
      break
  return layer_i_nodes

def _diffuse_k_rounds(G, A, steps):
  tried_edges = set()
  layer_i_nodes = [ ]
  layer_i_nodes.append([i for i in A])
  while steps > 0 and len(A) < len(G):
    len_old = len(A)
    (A, activated_nodes_of_this_round, cur_tried_edges) = \
        _diffuse_one_round(G, A, tried_edges)
    layer_i_nodes.append(activated_nodes_of_this_round)
    tried_edges = tried_edges.union(cur_tried_edges)
    if len(A) == len_old:
      break
    steps -= 1
  return layer_i_nodes

def _diffuse_one_round(G, A, tried_edges):
  activated_nodes_of_this_round = set()
  cur_tried_edges = set()
  for s in A:
    for nb in G.successors(s):
      if nb in A or (s, nb) in tried_edges or (s, nb) in cur_tried_edges:
        continue
      if _prop_success(G, s, nb):
        activated_nodes_of_this_round.add(nb)
      cur_tried_edges.add((s, nb))
  activated_nodes_of_this_round = list(activated_nodes_of_this_round)
  A.extend(activated_nodes_of_this_round)
  return A, activated_nodes_of_this_round, cur_tried_edges

def _prop_success(G, src, dest):
  return random.random() <= G[src][dest]['act_prob']



def graphInfecciones(infecciones):
  fig, ax = plt.subplots(1, 1)
 
  rects = ax.patches
  labels = [f's{x}' for x in range(len(infecciones))]
  ax.bar(numpy.arange(len(labels)),infecciones,align='center', alpha=0.5)
  plt.xticks(numpy.arange(len(labels)), labels)
  plt.show()

def mejorSeleccion(argv):
  if len(argv) < 2:
        sys.stderr.write("Usage: %s <input graph>\n" % (argv[0],))
        return 1

  graph_fn = argv[1]
  maxInfeccion=-1
  infecciones=[]
  semilla=[]
  maxSemillas=[]
  for i in range(30):
    infec,semillas = infeccion(graph_fn,20,3)
    infecciones.append(infec)
    semilla.append(semillas)
    if(infec> maxInfeccion):
      maxInfeccion=infec
      maxSemillas=semillas
  print(maxInfeccion,maxSemillas)
  print(infecciones)
  graphInfecciones(infecciones)
  
  return maxInfeccion,maxSemillas


def infeccion(archivo, numSemillas,numPasos):
    
    graph_fn = archivo
    G = nx.Graph()  #let's create the graph first
    G=nx.Graph(nx.read_pajek(graph_fn))
    Nodos = G.nodes()
    n = G.number_of_nodes() 
    semillas=random.choices(list(Nodos),k=numSemillas)
    diffusion = independent_cascade(G, semillas, steps = numPasos)

    infectados = [] # cantidad de infectados
    rataRec = 0.2 # tasa de recuperacion

    # creamos un solo paso de infeccion, para empezar a iterar desde un comienzo con una casacada de infectados.
    for i in diffusion:
        for j in i: infectados.append(j)

    tasaRec = 1 # indica cuantos agentes se debe recuperar por iteraion.
    diffu = []
    RecSuc = [] #Guarda los recuperados y no deja que se vulevan a infectar
    iteraciones=10
    while (iteraciones>0):
          diffu.append(infectados)
          #empezamos a recupear segun tasa 
          tasaRec = int(len(infectados)*rataRec)
          suceptibles = [] # cantidad de recuperados que pueden volver a infectarse
          for i in range(tasaRec):
              suceptibles.append(infectados.pop(random.randrange(len(infectados))))
          #controlamos los recuperados
          for i in suceptibles: RecSuc.append(i)
          #se vuelve y se coloca los nuevos focos de infeccion a la funcion infeccion.
          diffusion = independent_cascade(G, infectados, steps = 1)
          #diffu.append(suceptibles)
          infectados = []
          #algoritmo que exluye de los infectados los nodos recuperados.
          ControlRec = 0
          for i in diffusion:
              infectados.extend(i)

          Recuperados = []
          for i in RecSuc:
              if i not in Recuperados:
                 Recuperados.append(i)
          diffu.append(Recuperados)
          iteraciones-=1
    
    infectados = 0
    suceptil = 0
    recuperados = 0
    sanos = n     
    conts = 0
    infect = []
    sucep = []
    Recup = [0]
    tics = []
    infect.append(infectados)
    sucep.append(sanos)
    tics.append(conts)
    
    return len(diffu[-1]),semillas

def mainDibujo(argv):
    
    if len(argv) < 2:
        sys.stderr.write("Usage: %s <input graph>\n" % (argv[0],))
        return 1

    graph_fn = argv[1]
    G = nx.Graph()  #let's create the graph first
    G=nx.Graph(nx.read_pajek(graph_fn))
    Nodos = G.nodes()
    n = G.number_of_nodes() 
  
    diffusion = independent_cascade(G,['Oneill, M', 'Foster, I', 'Winful, H', 'Buchel, C', 'Selkov, E', 'Hoffman, A', 'Friedlander, G', 'Varian, H', 'Geisel, T', 'Shefi, O', 'Schrag, S', 'Mehring, C', 'Mel, B', 'Pothen, A', 'Narayan, V', 'Levy, W', 'Munoz, M', 'Karimipour, V', 'Carroll, T', 'Maghoul, F'], steps = 3)

    infectados = [] # cantidad de infectados
    rataRec = 0.2 # tasa de recuperacion

    # creamos un solo paso de infeccion, para empezar a iterar desde un comienzo con una casacada de infectados.
    for i in diffusion:
        for j in i: infectados.append(j)

    tasaRec = 1 # indica cuantos agentes se debe recuperar por iteraion.
    diffu = []
    RecSuc = [] #Guarda los recuperados y no deja que se vulevan a infectar
    iteraciones=10
    while (iteraciones>0):
          diffu.append(infectados)
          #empezamos a recupear segun tasa 
          tasaRec = int(len(infectados)*rataRec)
          suceptibles = [] # cantidad de recuperados que pueden volver a infectarse
          for i in range(tasaRec):
              suceptibles.append(infectados.pop(random.randrange(len(infectados))))
          #controlamos los recuperados
          for i in suceptibles: RecSuc.append(i)
          #se vuelve y se coloca los nuevos focos de infeccion a la funcion infeccion.
          diffusion = independent_cascade(G, infectados, steps = 1)
          #diffu.append(suceptibles)
          infectados = []
          #algoritmo que exluye de los infectados los nodos recuperados.
          ControlRec = 0
          for i in diffusion:
              infectados.extend(i)

          Recuperados = []
          for i in RecSuc:
              if i not in Recuperados:
                 Recuperados.append(i)
          diffu.append(Recuperados)
          iteraciones-=1

    
    pos=nx.fruchterman_reingold_layout(G, scale=1000)
  
 
    cf = plt.figure(2, figsize=(10,10))
    ax = cf.add_axes((0,0,1,1))
    
    labels={}
    cont = 1
    # for i in G.nodes():
    #     colr = float(cont)/n
    #     nx.draw_networkx_nodes(G, pos, [i] , node_size = 100, node_color = 'w')
    #     labels[i] = i
    #     cont += 1
    # nx.draw_networkx_labels(G,pos,labels,font_size=5)        
    # nx.draw_networkx_edges(G,pos, alpha=0.5)

    # plt.ion()
    # plt.draw()
    infectados = 0
    suceptil = 0
    recuperados = 0
    sanos = n     
    conts = 0
    infect = []
    sucep = []
    Recup = [0]
    tics = []
    infect.append(infectados)
    sucep.append(sanos)
    #Recup.append(suceptil)
    tics.append(conts)
    cont = 2
    for i in diffu:
        if cont % 2 == 0:
           infectados = len(i) 
          #  nx.draw_networkx_nodes(G, pos, i , node_size = 250, node_color = 'r')
          #  plt.pause(0.001)
          #  plt.draw()
        if cont % 2 != 0:
           suceptil = len(i)
           conts += 1
           tics.append(conts)
           sanos = n - infectados
           infect.append(infectados)
           sucep.append(sanos)
           Recup.append(suceptil)           
          #  nx.draw_networkx_nodes(G, pos, i , node_size = 300, node_color = 'b')
          #  plt.pause(0.001)
          #  plt.draw()
        cont += 1

    print ('infectados: ', infectados, ' sanos: ', sanos," Suceptibles: ", suceptil,  ' Etapas: ', conts)
 
    plt.subplot(2,2,1)
    plt.title('suceptibles')
    plt.xlabel('Tics')
    plt.ylabel('Nodos')
    plt.plot(tics,sucep,'r')
    plt.subplot(2,2,2)
    plt.title('infectados')
    plt.xlabel('Tics')
    plt.ylabel('Nodos')
    plt.plot(tics,infect,'g')
    plt.subplot(2,2,3)
    plt.title('infectados, sanos y Suceptibles')
    plt.xlabel('Tics')
    plt.ylabel('Nodos')
    plt.plot(tics,sucep)
    plt.plot(tics,Recup)
    plt.plot(tics,infect)
    plt.show()
    plt.subplot(2,2,4)
    plt.title('recuperados')
    plt.xlabel('Tics')
    plt.ylabel('Nodos')
    plt.plot(tics,Recup,'b')
   
    plt.pause(20)
    # Animator call
    #anim = animation.FuncAnimation(fig, animate, frames=20, interval=20, blit=True)
    # Creamos una figura
    blt.plot(infect,tics)
    blt.show()
    print (time.strftime("%I:%M:%S"))

if __name__ == "__main__":
    sys.exit(mainDibujo(sys.argv))
