# PUNTO 1 (20%). a. Simular las entradas y salidas de los
# vuelos de los aeropuertos de:
# New York JFK Internacional
# https://www.espanol.skyscanner.com/vuelos/llegadas-salidas/jfk/nueva-york-john-f-kennedy-llegadas-salidas
# Bogotá - El dorado
# https://eldorado.aero/vuelos/salidas
# Cali – Alfonso Bonilla
# https://www.aerocali.com.co/vuelos/informacion-de-vuelos/
# S2e tomar los datos de cada aeropuerto de los datos
# reales, y de allí sacar la distribución para simularlos.
# Tiempo estimado de simulación 24 horas.
# Tener en cuenta los VUELOS cancelados, retrasados y los
# que llegan y salen.
# En cada simulación identificar las partes de la misma,
# variables de estado, de desempeño, etc.
# b. Que escenarios debe se deben de hacer en los aeropuertos
# de Cali y Bogota para que queden a la par del aeropuerto de
# New York? Haga la simulación y de sus conclusiones.

# salidas 
# Ny
# uniforme
# bogota
# uniforme
# cali
# uniforme

# entradas
# NY
# exponencial
# bogota
# exponencial
# cali 
# uniforme

# VARIABLES DE ENTRADA
# AVION

# VARIABLES DE ESTADO
# ESTADO DE LA COLA DE DESPEGUE 
# ESTADO DE PUERTA DE ABORDAJE 
# ESTADO DEL AVION (VOLANDO, EN EL PISO,CANCELADO O ATRASADO)

# VARIABLES DE DESEMPEÑO 
# comportamiento de la cola 
# tamaño promedio de la cola
# tiempo promedio de espera en cola
# tiempo de llegada a destino final

import pandas as pd
import numpy
import matplotlib.pyplot as plt
from scipy.stats import uniform
from scipy.stats import expon

def infoNY():
    fechasSalida= [12,15,17,19,21]
    fechasEntrada= [12,14,16,18,20,22]
    datosSalida=[]
    datosEntrada=[]
    for fecha in fechasSalida:
        datosSalida.append(pd.read_html(f'https://es.airports-worldwide.info/aeropuerto/JFK/salidas/Salidas_Aeropuerto_Internacional_John_F_Kennedy_Nueva_York_JFK_KJFK?time=2022-02-28+{fecha}%3A00'))
    for fecha in fechasEntrada:
        datosEntrada.append(pd.read_html(f'https://es.airports-worldwide.info/aeropuerto/JFK/llegadas/Llegadas_Aeropuerto_Internacional_John_F_Kennedy_Nueva_York_JFK_KJFK?time=2022-02-28+{fecha}%3A00'))
    llegadas=[]
    salidas=[]
   
    for tablas in datosEntrada:
        for tabla in tablas:
            llegadas.extend(tabla['Llegada'])
    for tablas in datosSalida:
        for tabla in tablas:
            salidas.extend(tabla['Partida'])

    print(len(salidas),len(llegadas))
    return [salidas,llegadas]

def infoBogota():
    fechasSalida= [13,16,19]
    fechasEntrada= [12,15,18,20]
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
    return [salidas,llegadas]

def infoCali():
    
    datosSalida=[]
    datosEntrada=[]
    
    datosSalida=pd.read_html(f'https://es.airports-worldwide.info/aeropuerto/CLO/salidas/Salidas_Cali_Alfonso_Bonillaaragon_airport_Cali_CLO_SKCL')
    datosEntrada=pd.read_html(f'https://es.airports-worldwide.info/aeropuerto/CLO/llegadas/_Cali_Alfonso_Bonillaaragon_airport_Cali_CLO_SKCL')
    llegadas=[]
    salidas=[]
   
    for tabla in datosEntrada:
        # for tabla in tablas:
            llegadas.extend(tabla['Llegada'])
    for tabla in datosSalida:
        # for tabla in tablas:
            salidas.extend(tabla['Partida'])

    print(len(salidas),len(llegadas))
    return [salidas,llegadas]

def plot(horas):
    rango= [i for i in range(24)]
    plot1 = plt.figure(1)
    plt.hist(horas[0],bins=24,edgecolor='black', linewidth=1.2)
    plt.xlabel("horas de salida")
    plot2 = plt.figure(2)
    plt.hist(horas[1],bins=24,edgecolor='black', linewidth=1.2)
    plt.xlabel("horas de Entrada")
    plt.show()

def randExpon(size):
    return  [x * 24 for x in expon.rvs(size=size)]
def randUniform(size):
    return [x * 24 for x in uniform.rvs(size=size)]

# print(randUniform(5))
# plot(infoNY())
# plot(infoBogota())
# plot(infoCali())


#Datos de la simulación
SEMILLA = 4123410 #Semilla generador
AVIONES = 100 #Vamos a simular 10 clientes
# ATENCION_CLIENTES = [40, 100] #Clientes son atendidos en una distribucion

#Variables desempeño
tiempoLlegada=[]
COLA = 0
MAX_COLA = 0
ESPERA_AVIONES = numpy.array([])

def llegada(env, numero, contador,rand):
    for i in range(numero):
        c = avion(env, 'Avion %02d' % i, contador)
        env.process(c)
        tiempo_llegada = rand(1)
        yield env.timeout(tiempo_llegada) #Yield retorna un objeto iterable
        
def avion(env, nombre, servidor):
    #El cliente llega y se va cuando es atendido
    llegada = env.now
    print('%7.2f'%(env.now)," Llega el avion ", nombre)
    global COLA
    global MAX_COLA 
    global ESPERA_AVIONES   
    #Atendemos a los clientes (retorno del yield)
    #With ejecuta un iterador sin importar si hay excepciones o no
    with servidor.request() as req:
		
                #Hacemos la espera hasta que sea atendido el cliente
                COLA += 1
                if COLA > MAX_COLA:
                   MAX_COLA = COLA
		
                #print("Tamaño cola", COLA)
                results = yield req	
                COLA = COLA - 1
                espera = env.now - llegada
                ESPERA_AVIONES = numpy.append(ESPERA_AVIONES, espera)
		
                print('%7.2f'%(env.now), " El AVION ",nombre," espera a ser atendido ",espera)
		
                tiempo_atencion = random.uniform(ATENCION_CLIENTES[0],ATENCION_CLIENTES[1])
                yield env.timeout(tiempo_atencion)
		
                print('%7.2f'%(env.now), " Sale el cliente ",nombre)
    
                    
#Inicio de la simulación

print('Sala de cine')
random.seed(SEMILLA)
env = simpy.Environment()

#Inicio del proceso y ejecución
servidor = simpy.Resource(env, capacity=1)
env.process(llegada(env, CLIENTES, servidor))
env.run()

print("Cola máxima ",MAX_COLA)
print("Tiempo promedio de espera ",'%7.2f'%(numpy.mean(ESPERA_CLIENTES)))
