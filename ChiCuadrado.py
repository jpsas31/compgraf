import util
import math
from tabulate import tabulate

def calcularChiCuadrado(frecuencia_esperada, frecuencia_observada):
    return math.pow((frecuencia_esperada - frecuencia_observada),2)/frecuencia_esperada

def pruebaChiCuadrado(datos, confianza):
    numero_datos = len(datos)
    numero_clases = util.numeroClases(numero_datos)
    clases = util.crearClases(int(numero_clases))
    grados_libertad = numero_clases - 1

    frecuencias_esperadas = util.frecuenciasEsperadas(numero_datos, numero_clases)
    frecuencias_observadas = util.frecuenciasObservadas(clases, datos)

    chiCuadrados = list(map(lambda fe,fo: calcularChiCuadrado(fe,fo),frecuencias_esperadas,frecuencias_observadas))
    chiCuadrado = sum(chiCuadrados)
    print("GRADOS DE LIBERTAD = ", grados_libertad)
    #http://www.mat.uda.cl/hsalinas/cursos/2010/eyp2/Tabla%20Chi-Cuadrado.pdf
    estadistico = {
                    0.05: { 
                        9: 16.919,
                        8: 15.507,
                        17: 27.587,
                        18: 28.869,
                        21: 32.671,
                        19: 30.144

                    } 
                } 

    pasa = chiCuadrado <= estadistico[confianza][grados_libertad]

    res = []
    for fila in zip(clases,frecuencias_observadas, frecuencias_esperadas, chiCuadrados ):
         res.append(fila)
    return  [res, chiCuadrado, estadistico[confianza][grados_libertad], pasa]

def chiCuadradoTabla(datos, confianza):
    res = pruebaChiCuadrado(datos, confianza)
    print(f"X^2 = {res[1]}")
    print(f"X^2crit = {res[2]}")

    if(res[3]):
        print("El generador es bueno en cuanto a uniformidad pues el valor calculado es menor o igual al critico")
    else:
        print("El generador no es bueno en cuanto a uniformidad pues el valor calculado es mayor al critico")
    cabecera = ["clase", "FO", "FE", "(FE-FO)^2/FE"]
    print(tabulate(res[0], cabecera, tablefmt="grid"))

