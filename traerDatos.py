import pandas as pd

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