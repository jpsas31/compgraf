import numpy
import matplotlib.pyplot as plt
from scipy.stats import uniform
from scipy.stats import expon
import scipy.stats as stats
import util
import ChiCuadrado
import math
import random
import simpy

def plot(horas, intervalo_uno, intervalo_dos):
    plot1 = plt.figure(1)
    plt.hist(horas[0], bins=intervalo_uno,edgecolor='black', linewidth=1.2)
    plt.xlabel("horas de salida")
    plot2 = plt.figure(2)
    plt.hist(horas[1],bins=intervalo_dos, edgecolor='black', linewidth=1.2)
    plt.xlabel("horas de Entrada")
    plt.show()

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

cali[0] = [pasarSegundos(hora) for hora in cali[0]]
cali[1] = [pasarSegundos(hora) for hora in cali[1]]

bogota[0] = [pasarSegundos(hora) for hora in bogota[0]]
bogota[1] = [pasarSegundos(hora) for hora in bogota[1]]

newYork[0] = [pasarSegundos(hora) for hora in newYork[0]]
newYork[1] = [pasarSegundos(hora) for hora in newYork[1]]

# GRAFICA DATOS OBTENIDOS 
# plot(cali,math.ceil(math.sqrt(len(cali[0]))),math.ceil(math.sqrt(len(cali[1]))))
# plot(bogota, math.ceil(math.sqrt(len(bogota[0]))),math.ceil(math.sqrt(len(bogota[1]))))
# plot(newYork, math.ceil(math.sqrt(len(newYork[0]))),math.ceil(math.sqrt(len(newYork[1]))))


cali[0] = util.normalizarMedia(cali[0])
cali[1] = util.normalizarMedia(cali[1])

bogota[0] = util.normalizarMedia(bogota[0])
bogota[1] = util.normalizarMedia(bogota[1])

newYork[0] = util.normalizarMedia(newYork[0])
newYork[1] = util.normalizarMedia(newYork[1])

MEDIA_SALIDA_BG = util.media(bogota[0])
MEDIA_LLEGADA_BG = util.media(bogota[1])

MEDIA_SALIDA_NY = util.media(newYork[0])
MEDIA_LLEGADA_NY= util.media(newYork[1])

VARIANZA_SALIDA_BG = util.varianza(bogota[0], MEDIA_SALIDA_BG)
VARIANZA_LLEGADA_BG = util.varianza(bogota[1], MEDIA_LLEGADA_BG)

VARIANZA_SALIDA_NY = util.varianza(newYork[0], MEDIA_SALIDA_NY)
VARIANZA_LLEGADA_NY= util.varianza(newYork[1], MEDIA_LLEGADA_NY)


# PRUEBA DE CHICIADRADO CALI
print("Salidas Cali")
ChiCuadrado.chiCuadradoTabla(cali[0], 0.05)  
print("Llegadas Cali")
ChiCuadrado.chiCuadradoTabla(cali[1], 0.05)  


# PRUEBA DE NORMAL TEST BOGOTA - NY

alpha = 1e-20

k2, p = stats.normaltest(bogota[0])
print("BOGOTA LLEGADA: ","K2", k2, "P", p)

if(p < alpha):
    print("BOGOTA LLEGADA CUMPLE DISTRIBUCION NORMAL")

k2, p = stats.normaltest(bogota[0])
print("BOGOTA SALIDA: ","K2", k2, "P", p)


if(p < alpha):
    print("BOGOTA SALIDA CUMPLE DISTRIBUCION NORMAL")

k2, p = stats.normaltest(newYork[0])
print("NY LLEGADA: ","K2", k2, "P", p)

if(p < alpha):
    print("NY LLEGADA CUMPLE DISTRIBUCION NORMAL")

k2, p = stats.normaltest(newYork[0])
print("NY SALIDA: ","K2", k2, "P", p)

if(p < alpha):
    print("NY SALIDA CUMPLE DISTRIBUCION NORMAL")


# SIMULACION 

#Datos de la simulación
SEMILLA = 490

AVIONES_SALIDA =  30
AVIONES_LLEGADA = 30

LLEGADA_AVIONES = [0,45 + 10] #cada cuanto llega un avion que va a aterrizar se incluye el retrasp
LLEGADA_USO_PISTA = [10,30] #cuanto demora un avion que aterriza en usar la pista

SALIDA_AVIONES = [0,30 + 10] #cada cuanto llega un avion que va a despegar se incluye el retrasp
SALIDA_USO_PISTA = [10,30] #cuanto demora un avion en despegar


PROBABILIDAD_SALIDA_CANCELACIONES = 0.05 

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


def llegada(env, aviones, servidor, normal = False, mu = 0, sigma = 0):

    for i in range(aviones):

            c = cliente(env, 'avll%02d' % i, servidor, True)
            env.process(c)
            if(normal):
                tiempo_llegada = abs(random.normalvariate(mu, sigma)*55) 
            else:
                tiempo_llegada = random.uniform(LLEGADA_AVIONES[0],LLEGADA_AVIONES[1])
            yield env.timeout(tiempo_llegada)

def salida(env, aviones, servidor, normal = False, mu = 0, sigma = 0):

    for i in range(aviones):

            c = cliente(env, 'avss%02d' % i, servidor, False)
            env.process(c)
            if(normal):
                tiempo_salida = abs(random.normalvariate(mu, sigma)*30)
            else:
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

# #CALI
print('Areopuerto cali')
random.seed(SEMILLA)
env = simpy.Environment()
CANTIDAD_PISTAS = 2
AVIONES_SALIDA =  30
AVIONES_LLEGADA = 30
#Inicio del proceso y ejecución
servidor = simpy.Resource(env, capacity=CANTIDAD_PISTAS)
env.process(llegada(env, AVIONES_LLEGADA, servidor))
env.process(salida(env, AVIONES_SALIDA, servidor))
env.run()

#variables de desempeño
print("Cola máxima ",MAX_COLA)
print("Cola máxima aterrizar ",MAX_CANTIDAD_VUELOS_ATERRIZAR)
print("Cola máxima despegar ",MAX_CANTIDAD_VUELOS_DESPEGAR)
print("Tiempo de espera total",'%7.2f'%(numpy.sum(ESPERA_AVIONES)))
print("Tiempo de espera aviones por aterrizar",'%7.2f'%(numpy.sum(ESPERA_LLEGADA)))
print("Tiempo de espera aviones por despegar",'%7.2f'%(numpy.sum(ESPERA_SALIDA)))

print("Tiempo promedio de espera total",'%7.2f'%(numpy.mean(ESPERA_AVIONES)))
print("Tiempo promedio de espera aviones por aterrizar",'%7.2f'%(numpy.mean(ESPERA_LLEGADA)))
print("Tiempo promedio de espera aviones por despegar",'%7.2f'%(numpy.mean(ESPERA_SALIDA)))

#BOGOTA

CANTIDAD_VUELOS_ATERRIZAR = 0
CANTIDAD_VUELOS_DESPEGAR = 0

MAX_CANTIDAD_VUELOS_ATERRIZAR = 0
MAX_CANTIDAD_VUELOS_DESPEGAR = 0

COLA = 0
MAX_COLA = 0
ESPERA_AVIONES = numpy.array([])
ESPERA_LLEGADA = numpy.array([])
ESPERA_SALIDA = numpy.array([])


print('Areopuerto Bogota')
random.seed(SEMILLA)
env = simpy.Environment()
CANTIDAD_PISTAS = 2
AVIONES_SALIDA =  30
AVIONES_LLEGADA = 30
#Inicio del proceso y ejecución
servidor = simpy.Resource(env, capacity=CANTIDAD_PISTAS)
env.process(llegada(env, AVIONES_LLEGADA, servidor, True, MEDIA_LLEGADA_BG, math.sqrt(VARIANZA_LLEGADA_BG)))
env.process(salida(env, AVIONES_SALIDA, servidor, True, MEDIA_SALIDA_BG, math.sqrt(VARIANZA_SALIDA_BG)))
env.run()

#variables de desempeño
print("Cola máxima ",MAX_COLA)
print("Cola máxima aterrizar ",MAX_CANTIDAD_VUELOS_ATERRIZAR)
print("Cola máxima despegar ",MAX_CANTIDAD_VUELOS_DESPEGAR)
print("Tiempo de espera total",'%7.2f'%(numpy.sum(ESPERA_AVIONES)))
print("Tiempo de espera aviones por aterrizar",'%7.2f'%(numpy.sum(ESPERA_LLEGADA)))
print("Tiempo de espera aviones por despegar",'%7.2f'%(numpy.sum(ESPERA_SALIDA)))

print("Tiempo promedio de espera total",'%7.2f'%(numpy.mean(ESPERA_AVIONES)))
print("Tiempo promedio de espera aviones por aterrizar",'%7.2f'%(numpy.mean(ESPERA_LLEGADA)))
print("Tiempo promedio de espera aviones por despegar",'%7.2f'%(numpy.mean(ESPERA_SALIDA)))

# #NEWYORK

CANTIDAD_VUELOS_ATERRIZAR = 0
CANTIDAD_VUELOS_DESPEGAR = 0

MAX_CANTIDAD_VUELOS_ATERRIZAR = 0
MAX_CANTIDAD_VUELOS_DESPEGAR = 0

COLA = 0
MAX_COLA = 0
ESPERA_AVIONES = numpy.array([])
ESPERA_LLEGADA = numpy.array([])
ESPERA_SALIDA = numpy.array([])


print('Areopuerto NY')
random.seed(SEMILLA)
env = simpy.Environment()
CANTIDAD_PISTAS = 2
AVIONES_SALIDA =  30
AVIONES_LLEGADA = 30
#Inicio del proceso y ejecución
servidor = simpy.Resource(env, capacity=CANTIDAD_PISTAS)
env.process(llegada(env, AVIONES_LLEGADA, servidor, True, MEDIA_LLEGADA_NY, math.sqrt(VARIANZA_LLEGADA_NY)))
env.process(salida(env, AVIONES_SALIDA, servidor, True, MEDIA_SALIDA_NY, math.sqrt(VARIANZA_SALIDA_NY)))
env.run()

#variables de desempeño
print("Cola máxima ",MAX_COLA)
print("Cola máxima aterrizar ",MAX_CANTIDAD_VUELOS_ATERRIZAR)
print("Cola máxima despegar ",MAX_CANTIDAD_VUELOS_DESPEGAR)
print("Tiempo de espera total",'%7.2f'%(numpy.sum(ESPERA_AVIONES)))
print("Tiempo de espera aviones por aterrizar",'%7.2f'%(numpy.sum(ESPERA_LLEGADA)))
print("Tiempo de espera aviones por despegar",'%7.2f'%(numpy.sum(ESPERA_SALIDA)))

print("Tiempo promedio de espera total",'%7.2f'%(numpy.mean(ESPERA_AVIONES)))
print("Tiempo promedio de espera aviones por aterrizar",'%7.2f'%(numpy.mean(ESPERA_LLEGADA)))
print("Tiempo promedio de espera aviones por despegar",'%7.2f'%(numpy.mean(ESPERA_SALIDA)))
