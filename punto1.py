import pandas as pd
import numpy
import matplotlib.pyplot as plt
from scipy.stats import uniform
from scipy.stats import expon
import util
import ChiCuadrado
import math
import random
import simpy

def infoNY():
    fechasSalida= [12,14,16,18,20,22]
    fechasEntrada= [12,14,16,18,20,22]
    datosSalida=[]
    datosEntrada=[]
    for fecha in fechasSalida:
        datosSalida.append(pd.read_html(f'https://es.airports-worldwide.info/aeropuerto/JFK/salidas/Salidas_Aeropuerto_Internacional_John_F_Kennedy_Nueva_York_JFK_KJFK?time=2022-03-13+{fecha}%3A00'))
    for fecha in fechasEntrada:
        datosEntrada.append(pd.read_html(f'https://es.airports-worldwide.info/aeropuerto/JFK/llegadas/Llegadas_Aeropuerto_Internacional_John_F_Kennedy_Nueva_York_JFK_KJFK?time=2022-03-13+{fecha}%3A00'))
    llegadas=[]
    salidas=[]
   
    for tablas in datosEntrada:
        for tabla in tablas:
            llegadas.extend(tabla['Llegada'])
    for tablas in datosSalida:
        for tabla in tablas:
            salidas.extend(tabla['Partida'])

    print(len(salidas),len(llegadas))
    with open('ny_l.txt', 'w') as sal :
        sal.write(' '.join(salidas))

    with open('ny_s.txt', 'w') as sal :
        sal.write(' '.join(llegadas))

    return [salidas,llegadas]

def infoBogota():
    fechasSalida= [11,14,17, 20, 23]
    fechasEntrada= [11, 14, 17, 21]
    datosSalida=[]
    datosEntrada=[]
    for fecha in fechasSalida:
        datosSalida.append(pd.read_html(f'https://es.airports-worldwide.info/aeropuerto/BOG/salidas/Salidas_Aeropuerto_El_Dorado_Bogota_BOG_SKBO?time=2022-02-28+{fecha}%3A00'))
    for fecha in fechasEntrada:
        datosEntrada.append(pd.read_html(f'https://es.airports-worldwide.info/aeropuerto/BOG/llegadas/Llegadas_Aeropuerto_El_Dorado_Bogota_BOG_SKBO?time=2022-02-28+{fecha}%3A00'))
    llegadas=[]
    salidas=[]
   
    for tablas in datosEntrada:
        for tabla in tablas:
            llegadas.extend(tabla['Llegada'])
    for tablas in datosSalida:
        for tabla in tablas:
            salidas.extend(tabla['Partida'])

    print(len(salidas),len(llegadas))
    with open('bg_l.txt', 'w') as sal :
        sal.write(' '.join(salidas))

    with open('bg_s.txt', 'w') as sal :
        sal.write(' '.join(llegadas))
    return [salidas,llegadas]

def infoCali():
    
    datosSalida=[]
    datosEntrada=[]
    
    datosSalida=pd.read_html(f'https://es.airports-worldwide.info/aeropuerto/CLO/salidas/Salidas_Cali_Alfonso_Bonillaaragon_airport_Cali_CLO_SKCL')
    datosEntrada=pd.read_html(f'https://es.airports-worldwide.info/aeropuerto/CLO/llegadas/_Cali_Alfonso_Bonillaaragon_airport_Cali_CLO_SKCL')
    llegadas=[]
    salidas=[]
   
    for tabla in datosEntrada:
            llegadas.extend(tabla['Llegada'])
    for tabla in datosSalida:
            salidas.extend(tabla['Partida'])

    print(len(salidas),len(llegadas))
    with open('ca_l.txt', 'w') as sal :
        sal.write(' '.join(salidas))

    with open('ca_s.txt', 'w') as sal :
        sal.write(' '.join(llegadas))
    return [salidas,llegadas]

def plot(horas, intervalo_uno, intervalo_dos):
    plot1 = plt.figure(1)
    plt.hist(horas[0], bins=intervalo_uno,edgecolor='black', linewidth=1.2)
    plt.xlabel("horas de salida")
    plot2 = plt.figure(2)
    plt.hist(horas[1],bins=intervalo_dos, edgecolor='black', linewidth=1.2)
    plt.xlabel("horas de Entrada")
    plt.show()

def randExpon(size):
    return  [x * 24 for x in expon.rvs(size=size)]
def randUniform(size):
    return [x * 24 for x in uniform.rvs(size=size)]

def pasarSegundos(hora):
    horas = hora.split(":")
    return 3600*int(horas[0]) + int(horas[1])*60


#Obtenemos los datos:
# primera lista es salida
# segunda lista es llegada
cali = [[],[]]
bogota =[[],[]]
newYork = [[],[]]

with open('./ca_s.txt', 'r') as sal :
       lin = sal.read().split(' ')
       cali[0] = lin
with open('./ca_l.txt', 'r') as sal :
       lin = sal.read().split(' ')
       cali[1] = lin

with open('./ny_s.txt', 'r') as sal :
       lin = sal.read().split(' ')
       newYork[0] = lin 
with open('./ny_l.txt', 'r') as sal :
       lin = sal.read().split(' ')
       newYork[1] = lin 

with open('./bg_s.txt', 'r') as sal :
       lin = sal.read().split(' ')
       bogota[0] = lin 
with open('./bg_l.txt', 'r') as sal :
       lin = sal.read().split(' ')
       bogota[1] = lin 

# print(cali, bogota, newYork)


# print(randUniform(5))
# plot(cali)   #uniforme
# plot(bogota) #uniforme
# plot(newYork) #uniforme - normal

# plot(cali,math.ceil(math.sqrt(len(cali[0]))),math.ceil(math.sqrt(len(cali[1]))))
# plot(bogota)
# plot(newYork)
cali[0] = [pasarSegundos(hora) for hora in cali[0]]
cali[1] = [pasarSegundos(hora) for hora in cali[1]]

bogota[0] = [pasarSegundos(hora) for hora in bogota[0]]
bogota[1] = [pasarSegundos(hora) for hora in bogota[1]]

newYork[0] = [pasarSegundos(hora) for hora in newYork[0]]
newYork[1] = [pasarSegundos(hora) for hora in newYork[1]]


# plot(cali,math.ceil(math.sqrt(len(cali[0]))),math.ceil(math.sqrt(len(cali[1]))))
# plot(bogota, math.ceil(math.sqrt(len(bogota[0]))),math.ceil(math.sqrt(len(bogota[1]))))
# plot(newYork, math.ceil(math.sqrt(len(newYork[0]))),math.ceil(math.sqrt(len(newYork[1]))))
print("Salidas Cali")
ChiCuadrado.chiCuadradoTabla(util.normalizarMedia(cali[0]), 0.05)  
print("Llegadas Cali")
ChiCuadrado.chiCuadradoTabla(util.normalizarMedia(cali[1]), 0.05)  

print("Salidas Bogota")
ChiCuadrado.chiCuadradoTabla(util.normalizarMedia(bogota[0]), 0.05)  
print("Llegadas Bogota")
ChiCuadrado.chiCuadradoTabla(util.normalizarMedia(bogota[1]), 0.05)  

print("Salidas NewYork")
ChiCuadrado.chiCuadradoTabla(util.normalizarMedia(newYork[0]), 0.05) 
print("Llegadas NewYork") 
ChiCuadrado.chiCuadradoTabla(util.normalizarMedia(newYork[1]), 0.05)  



# plot([util.normalizarMedia(cali[0]),util.normalizarMedia(cali[1])],math.ceil(math.sqrt(len(cali[0]))),math.ceil(math.sqrt(len(cali[1]))) )
# plot([util.normalizarMedia(bogota[0]),util.normalizarMedia(bogota[1])], math.ceil(math.sqrt(len(bogota[0]))),math.ceil(math.sqrt(len(bogota[1]))))
# plot([util.normalizarMedia(newYork[0]),util.normalizarMedia(newYork[1])],math.ceil(math.sqrt(len(newYork[0]))),math.ceil(math.sqrt(len(newYork[1]))))


# SIMULACION 1

# VARIABLES DE ENTRADA
# Tiempo de llegada
# Tiempo de salida
# Retrasos
# Cancelaciones

#Datos de la simulación
SEMILLA = 490

AVIONES_SALIDA =  30
AVIONES_LLEGADA = 30

LLEGADA_AVIONES = [0,45 + 10] #cada cuanto llega un avion que va a aterrizar se incluye el retrasp
LLEGADA_USO_PISTA = [10,30] #cuanto demora un avion que aterriza en usar la pista

SALIDA_AVIONES = [0,30 + 10] #cada cuanto llega un avion que va a despegar se incluye el retrasp
SALIDA_USO_PISTA = [10,30] #cuanto demora un avion en despegar


PROBABILIDAD_SALIDA_CANCELACIONES = 0.05 

#Variables de estado


#Variables de desempeño

CANTIDAD_VUELOS_ATERRIZAR = 0
CANTIDAD_VUELOS_DESPEGAR = 0

MAX_CANTIDAD_VUELOS_ATERRIZAR = 0
MAX_CANTIDAD_VUELOS_DESPEGAR = 0

COLA = 0
MAX_COLA = 0
ESPERA_AVIONES = numpy.array([])
ESPERA_LLEGADA = numpy.array([])
ESPERA_SALIDA = numpy.array([])


def llegada(env, aviones, servidor):

    for i in range(aviones):
            c = cliente(env, 'avll%02d' % i, servidor, True)
            env.process(c)
            tiempo_llegada = random.uniform(LLEGADA_AVIONES[0],LLEGADA_AVIONES[1])
            yield env.timeout(tiempo_llegada)

def salida(env, aviones, servidor):

    for i in range(aviones):
            c = cliente(env, 'avss%02d' % i, servidor, False)
            env.process(c)
            tiempo_salida = random.uniform(SALIDA_AVIONES[0],SALIDA_AVIONES[1])
            yield env.timeout(tiempo_salida)

def cliente(env, nombre, servidor, llega):

    if(llega):
        mensaje = "Llega para aterrizar"
    else:
        mensaje = "Llega para despegar"        
        
    #El avion llega y se va cuando aterriza
    llegada = env.now
    print('%7.2f'%(env.now),f"{mensaje} el avion ", nombre)
    global COLA
    global MAX_COLA 
    global ESPERA_AVIONES
    global ESPERA_LLEGADA
    global ESPERA_SALIDA   
    global CANTIDAD_VUELOS_ATERRIZAR 
    global CANTIDAD_VUELOS_DESPEGAR 
    global MAX_CANTIDAD_VUELOS_ATERRIZAR 
    global MAX_CANTIDAD_VUELOS_DESPEGAR 
    global PROBABILIDAD_SALIDA_CANCELACIONES
    #cuanto demora en aterrizar o despegar el avion (retorno del yield)
    #With ejecuta un iterador sin importar si hay excepciones o no
    with servidor.request() as req:
                #calcular la prob
                
                #Hacemos la espera hasta que pueda aterrizar o despegar el avion
                COLA += 1
                if COLA > MAX_COLA:
                   MAX_COLA = COLA
                
                if(llega):
                    CANTIDAD_VUELOS_ATERRIZAR += 1
                    if CANTIDAD_VUELOS_ATERRIZAR > MAX_CANTIDAD_VUELOS_ATERRIZAR:
                        MAX_CANTIDAD_VUELOS_ATERRIZAR = CANTIDAD_VUELOS_ATERRIZAR
                else:
                    CANTIDAD_VUELOS_DESPEGAR += 1
                    if CANTIDAD_VUELOS_DESPEGAR > MAX_CANTIDAD_VUELOS_DESPEGAR:
                        MAX_CANTIDAD_VUELOS_DESPEGAR = CANTIDAD_VUELOS_DESPEGAR
		
                results = yield req	
                COLA = COLA - 1
                if(llega):
                    CANTIDAD_VUELOS_ATERRIZAR -= 1
                else:
                    CANTIDAD_VUELOS_DESPEGAR -= 1

                if(random.uniform(0,1) <= PROBABILIDAD_SALIDA_CANCELACIONES ):
                    espera = 0
                    ESPERA_AVIONES = numpy.append(ESPERA_AVIONES, espera)
                    yield env.timeout(espera)
                    print('%7.2f'%(env.now), f"se cancelo el avion ",nombre)
                else:
                    espera = env.now - llegada
                    ESPERA_AVIONES = numpy.append(ESPERA_AVIONES, espera)
                    if(llega):
                        ESPERA_LLEGADA = numpy.append(ESPERA_LLEGADA, espera)
                        mensaje = "aterrizar"
                    else:
                        ESPERA_SALIDA = numpy.append(ESPERA_SALIDA, espera) 
                        mensaje = "despegar"
                    
                    print('%7.2f'%(env.now), " El avion ",nombre,f"espera la pista para poder {mensaje}",espera)
		
                    if(llega):
                        tiempo_atencion = random.uniform(LLEGADA_USO_PISTA[0],LLEGADA_USO_PISTA[1])
                        mensaje = "aterrizo"
                    else:
                        tiempo_atencion = random.uniform(SALIDA_USO_PISTA[0],SALIDA_USO_PISTA[1])
                        mensaje = "despego"
                    yield env.timeout(tiempo_atencion)
                    print('%7.2f'%(env.now), f"{mensaje} el avion ",nombre)

                
    
#Inicio de la simulación

print('Areopuerto cali')
random.seed(SEMILLA)
env = simpy.Environment()

#escenarios
CANTIDAD_PISTAS = 1
#Inicio del proceso y ejecución
servidor = simpy.Resource(env, capacity=CANTIDAD_PISTAS)
env.process(llegada(env, AVIONES_LLEGADA, servidor))
env.process(salida(env, AVIONES_SALIDA, servidor))
env.run()


#desempeño
print("Cola máxima ",MAX_COLA)
print("Cola máxima aterrizar ",MAX_CANTIDAD_VUELOS_ATERRIZAR)
print("Cola máxima despegar ",MAX_CANTIDAD_VUELOS_DESPEGAR)
print("Tiempo de espera total",'%7.2f'%(numpy.sum(ESPERA_AVIONES)))
print("Tiempo de espera aviones por aterrizar",'%7.2f'%(numpy.sum(ESPERA_LLEGADA)))
print("Tiempo de espera aviones por despegar",'%7.2f'%(numpy.sum(ESPERA_SALIDA)))

print("Tiempo promedio de espera total",'%7.2f'%(numpy.mean(ESPERA_AVIONES)))
print("Tiempo promedio de espera aviones por aterrizar",'%7.2f'%(numpy.mean(ESPERA_LLEGADA)))
print("Tiempo promedio de espera aviones por despegar",'%7.2f'%(numpy.mean(ESPERA_SALIDA)))



