#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alejandro Naranjo Caraza
000190984
Created on Fri Dec 13 02:07:28 2024
Referencias
    ##https://medium.com/@abelkrw/exploring-heuristics-analytics-with-python-31b5c6e7e526
    ##Metaheuristics: From Design to Implementation (Wiley Series on Parallel and Distributed Computing)
    ##Introduccion a la investigacion de operaciones
"""

import numpy as np


# Consideramos a cada índice como un nodo
# Las trayectorias serán expresadas como listas de la forma t=[v1,v2,...,vn,v1]
# Nótese que no se pueden repetir aristas v_k --> v_{k+1} y todo nodo debe estar
#   incluido en t.
# Definimos funciones
# definimos objecto para matriz de distanciasy calculo de distancias totales de trayectorias
class Distancias:
    def __init__(self, path):
        try:
            coord_mat = np.loadtxt(path, usecols=(1, 2), dtype=float)
        except ValueError as e:
            raise ValueError(f"Error al leer el archivo {path}: {e}")
        if coord_mat.ndim != 2 or coord_mat.shape[1] != 2:
            raise ValueError("Error. Config de archivo.")
        dist_mat = np.linalg.norm(coord_mat[:, np.newaxis] - coord_mat, axis=2)
        dist_mat = np.round(dist_mat).astype(int)
        if not isinstance(dist_mat, np.ndarray) or not np.allclose(
            dist_mat, dist_mat.T
        ):
            raise ValueError("Debe ser matriz simétrica.")
        elif not dist_mat.dtype == int:
            raise ValueError("Debe ser matriz de enteros.")
        else:
            self.dist_mat = dist_mat
            self.size = dist_mat.shape[0]

    def distancia_total(
        self, t
    ):  # calcula distanica total recorrida por una trayectoria
        if not isinstance(t, list):
            raise TypeError("Error. Dtype trayectoria.")
        if len(t) - 1 != self.size:
            print(len(t))
            raise ValueError("Error. Tamaño de trayectoria.")
        res = sum(self.dist_mat[t[i], t[i - 1]] for i in range(1, len(t)))
        return res


def generar_poblacion(num_poblacion, num_ciudades):
    if type(num_poblacion) is not int or type(num_ciudades) is not int:
        print("Error. Inputs enteros para generar población.")
        res = None
    elif num_poblacion <= 0 or num_ciudades <= 0:
        print("Error. Enteros mayores a cero para generar población.")
        res = None
    else:
        poblacion = []
        for i in range(num_poblacion):
            t = list(np.random.permutation(num_ciudades))
            t.append(t[0])
            poblacion.append(t)
        res = poblacion
    return res


# Funciones de enlace
def single_point(t1, t2):  # single point crossover
    if not (isinstance(t1, list) and isinstance(t2, list)):
        print("Error. Dtype de trayectorias.")
        res = None
    elif len(t1) != len(t2):
        print("Error. Trayectorias de distinta longitud.")
        res = None
    elif len(t1) == 0 or len(t2) == 0:
        print("Error. Trayectorias de longitud 0.")
        res = None
    else:
        punto_enlace = np.random.randint(0, len(t1))
        hijo1 = t1[:punto_enlace]
        hijo2 = t2[:punto_enlace]
        for i in t2:
            if i not in hijo1:
                hijo1.append(i)
        for i in t1:
            if i not in hijo2:
                hijo2.append(i)
        hijo1.append(hijo1[0])
        hijo2.append(hijo2[0])
        res = hijo1, hijo2
    return res


def order_crossover(t1, t2):  # order crossover
    if not (isinstance(t1, list) and isinstance(t2, list)):
        print("Error. Dtype de trayectorias.")
        res = None
    elif len(t1) != len(t2):
        print("Error. Trayectorias de distinta longitud.")
        res = None
    elif len(t1) == 0 or len(t2) == 0:
        print("Error. Trayectorias de longitud 0.")
        res = None
    else:
        t1_temp = t1[:-1]
        t2_temp = t2[:-1]
        n = len(t1_temp)
        punto_enlace1 = np.random.randint(1, n // 2)
        punto_enlace2 = np.random.randint(n // 2, n - 1)
        hijo1, hijo2 = [-1] * n, [-1] * n
        hijo1[punto_enlace1:punto_enlace2] = t1_temp[punto_enlace1:punto_enlace2]
        hijo2[punto_enlace1:punto_enlace2] = t2_temp[punto_enlace1:punto_enlace2]

        def llenar_hijo(hijo, padre, punto_enlace2):
            indice = punto_enlace2 % n
            for i in padre:
                if i not in hijo:
                    hijo[indice] = i
                    indice = (indice + 1) % n
            return hijo

        hijo1 = llenar_hijo(hijo1, t2_temp, punto_enlace2)
        hijo2 = llenar_hijo(hijo2, t1_temp, punto_enlace2)
        hijo1.append(hijo1[0])
        hijo2.append(hijo2[0])
        res = hijo1, hijo2
    return res


def pmx_crossover(t1, t2):
    if not (isinstance(t1, list) and isinstance(t2, list)):
        print("Error. Dtype de trayectorias.")
        res = None
    elif len(t1) != len(t2):
        print("Error. Trayectorias de distinta longitud.")
        res = None
    elif len(t1) == 0 or len(t2) == 0:
        print("Error. Trayectorias de longitud 0.")
        res = None
    else:
        t1_temp = t1[:-1]
        t2_temp = t2[:-1]
        n = len(t1_temp)
        punto_enlace = np.random.randint(0, n)
        hijo1 = t1_temp[:punto_enlace] + t2_temp[punto_enlace:]
        hijo2 = t2_temp[:punto_enlace] + t1_temp[punto_enlace:]

        def reparar_hijo(hijo, padre):
            hijo_set = set(hijo)
            faltantes = [i for i in padre if i not in hijo_set]
            temp = set()
            for i in range(len(hijo)):
                if hijo[i] in temp:
                    hijo[i] = faltantes.pop()
                else:
                    temp.add(hijo[i])
            return hijo

        hijo1 = reparar_hijo(hijo1, t2_temp)
        hijo2 = reparar_hijo(hijo2, t1_temp)
        hijo1.append(hijo1[0])
        hijo2.append(hijo2[0])
        res = hijo1, hijo2
    return res


def mutacion(t, proporcion_mutaciones):
    if not isinstance(t, list):
        print("Error. Dtype de trayectoria.")
        res = None
    elif len(t) == 0:
        print("Error. Trayectoria de longitud 0.")
        res = None
    else:
        for i in range(1, len(t) - 1):
            if np.random.rand() < proporcion_mutaciones:
                k = np.random.randint(1, len(t) - 1)
                t[k], t[i] = t[i], t[k]
        res = t
    return res


def algoritmo_genetico(
    distancias, num_poblacion, iteraciones, proporcion_mutaciones, enlace
):
    if not isinstance(distancias, Distancias):
        print("Error. Dtype Distancias.")
        return None
    elif not (
        isinstance(num_poblacion, int)
        and isinstance(iteraciones, int)
        and isinstance(proporcion_mutaciones, float)
    ):
        print("Error. Dtype")
        return None
    else:
        num_ciudades = len(distancias.dist_mat)
        poblacion = generar_poblacion(num_poblacion, num_ciudades)
        ordenacion = lambda x: distancias.distancia_total(x)
        for i in range(iteraciones):
            print("Generación", i + 1)  ############################################
            poblacion = sorted(poblacion, key=ordenacion)
            poblacion_siguiente = poblacion[: num_poblacion // 2]
            while len(poblacion_siguiente) < num_poblacion:
                indice1, indice2 = np.random.choice(len(poblacion), 2, replace=False)
                t1, t2 = poblacion[indice1], poblacion[indice2]
                hijo1, hijo2 = enlace(t1, t2)
                hijo1, hijo2 = (
                    mutacion(hijo1, proporcion_mutaciones),
                    mutacion(hijo2, proporcion_mutaciones),
                )
                poblacion_siguiente.extend([hijo1, hijo2])
            poblacion = poblacion_siguiente
        res = poblacion[0]
    return res


# Importamos datos y generamos matriz de distancias
def principal(
    num_poblacion, generaciones, proporcion_mutaciones, enlace, path="zimbabwe.txt"
):
    distancias = Distancias(path)
    trayectoria_optima = algoritmo_genetico(
        distancias, num_poblacion, generaciones, proporcion_mutaciones, enlace
    )
    return trayectoria_optima, distancias.distancia_total(trayectoria_optima)


num_poblacion = 100
generaciones = 50
proporcion_mutaciones = 0.05
trayectoria_optima, distancia_optima = principal(
    num_poblacion, generaciones, proporcion_mutaciones, order_crossover
)

