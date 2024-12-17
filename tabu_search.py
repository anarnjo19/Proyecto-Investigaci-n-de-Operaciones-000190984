#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alejandro Naranjo Caraza
000190984
Created on Sat Dec 14 00:28:32 2024
"""

from collections import deque
import numpy as np
import copy
import matplotlib.pyplot as plt


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

    def distancia_total(self, t):
        if not isinstance(t, list):
            raise TypeError("Error. Dtype trayectoria.")
        if len(t) - 1 != self.size:
            raise ValueError("Error. Tamaño de trayectoria.")
        res = sum(self.dist_mat[t[i], t[i - 1]] for i in range(1, len(t)))
        return res


def generar_trayectoria(num_ciudades):
    if type(num_ciudades) is not int:
        temp = False
    elif num_ciudades <= 0:
        temp = False
    else:
        temp = True
    if temp:
        t = list(np.random.permutation(num_ciudades))
        t.append(t[0])
        res = t
    else:
        print("Error. Generación de trayectoria.")
        res = None
    return res


def subviaje_optimo(t, distancias, tabu):
    if not isinstance(t, list):
        res = None
    elif len(t) == 0:
        res = None
        print("Error en en inputs de subviaje óptimo.")
    elif not isinstance(distancias, Distancias):
        res = None
        print("Error en en inputs de subviaje óptimo.")
    elif not isinstance(tabu, deque):
        res = None
        print("Error en en inputs de subviaje óptimo.")
    else:
        subviaje_optimo = None
        distancia_optima = distancias.distancia_total(t)
        n = len(t)
        for i in range(1, n - 1):
            print(i)
            for j in range(i + 1, n - 1):
                t_sub = copy.copy(t)
                t_sub[i:j] = t[i:j][::-1]
                lig_x_eliminar1 = {t[i - 1], t[i]}
                lig_x_eliminar2 = {t[j], t[j + 1]}
                dist = distancias.distancia_total(t_sub)
                if (
                    dist < distancia_optima
                    and lig_x_eliminar1 not in tabu
                    and lig_x_eliminar2 not in tabu
                ):
                    distancia_optima = dist
                    subviaje_optimo = t_sub
                    lig_agregada1 = {t[i - 1], t[j]}
                    lig_agregada2 = {t[j + 1], t[i]}
                    tabu.append(lig_agregada1)
                    tabu.append(lig_agregada2)
        res = subviaje_optimo
    return res


def tabu_search(distancias, iteraciones, dec_s):
    if not isinstance(distancias, Distancias) or type(iteraciones) is not int:
        res = None
        print("Error. Tabu_search.")
    elif iteraciones <= 0:
        res = None
        print("Error. Tabu_search.")
    else:
        tabu = deque(maxlen=4)
        t = generar_trayectoria(distancias.dist_mat.shape[0])
        t_optima = t
        distancia_optima = distancias.distancia_total(t)
        for i in range(iteraciones):
            t_sub = subviaje_optimo(t, distancias, tabu)
            if t_sub == None:
                break
            t = t_sub
            dist = distancias.distancia_total(t)
            if dist < distancia_optima:
                distancia_optima = dist
                t_optima = t
        res = t_optima, distancia_optima
    return res


path = "zimbabwe.txt"
distancias = Distancias(path)
indices = []
obs = []
for i in range(1, 6):
    t_optima, distancia_optima = tabu_search(distancias, 2, int(2 * i))
    obs.append(distancia_optima)
    indices.append(int(2 * i))

plt.plot(indices, obs, marker="o", markersize=5)

plt.title("Distancias obtenidas vs tamaño de lista tabú")
plt.xlabel("Tamaño de lista tabú")
plt.ylabel("Distancias")
plt.savefig("tabu.jpg", dpi=300)

plt.show()

