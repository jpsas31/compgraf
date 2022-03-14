import math

#retorna el numero mas grande de una lista de numeros
def max(datos):
    max = datos[0]
    for num in datos:
        if(num >= max):
            max = num
    return max

#normalizacion de numero al rango [0,1]
def normalizar(datos):
    datos_normalizados = []
    maximo = max(datos)
    for dato in datos:
        datos_normalizados.append(dato/maximo)
    return datos_normalizados

def normalizarMedia(datos): 
    datos_normalizados = []
    maximo = max(datos)
    minimo = min(datos)
    for dato in datos:
        datos_normalizados.append((dato - minimo) /(maximo - minimo))
    return datos_normalizados
#retorna el numero de clases al realizar el techo de raiz cuadrada del numero de datos
def numeroClases(numero_datos):
    return math.ceil(math.sqrt(numero_datos))

#retorna la frecuencia observada de datos en un intervalo (limInferior, limSuperior]
def frecuenciaObservada(limiteInferior, limiteSuperior, datos):
    frecuencia = 0
    for dato in datos:
        if(dato > limiteInferior and dato <= limiteSuperior):
            frecuencia += 1
    return frecuencia

#retorna una lista de las frecuencias observadas para cada intervalo que se encuentra en una lista
def frecuenciasObservadas(intervalos, datos):
    frecuencias = []
    for intervalo in intervalos:
        frecuencias.append(frecuenciaObservada(intervalo[0], intervalo[1], datos))
    return frecuencias

#retorna una lista de clases (intervalos)
def crearClases(clases):
    longitud_intervalo = 1/clases
    intervalos = []
    for i in range(clases):
        intervalos.append([longitud_intervalo*i, longitud_intervalo*(i+1)])
    return intervalos

#retorna una lista de las frecuencias esperadas para una distribucion uniforme con x cantidad de clases y w numero de datos
def frecuenciasEsperadas(numero_datos, numero_clases):
    frec_esperada = numero_datos/numero_clases
    frecuenciaEperada = []
    for i in range(numero_clases):
        frecuenciaEperada.append(frec_esperada)
    return frecuenciaEperada

#retorna una lista de los datos acumulados para una lista de datos numericos
def acumula(datos):
    datos_acumulados = []
    for index, dato in enumerate(datos):
        if(index == 0):
            datos_acumulados.append(dato)
            continue
        datos_acumulados.append(dato+datos_acumulados[index-1])
    return datos_acumulados
    