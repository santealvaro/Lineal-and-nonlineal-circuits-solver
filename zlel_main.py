#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. module:: zlel_main.py
    :synopsis: Put yours

.. moduleauthor:: Put yours

"""
import zlel.zlel_p1 as zl1
import zlel.zlel_p2 as zl2
import zlel.zlel_p3 as zl3
import zlel.zlel_p4 as zl4
import numpy as np
import sys
import time

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "cirs/all/adibide_klase.cir"
    [cir_el, cir_nd, cir_val, cir_ctr, funtzioa] = zl1.cir_parser(filename)
    rows, cols = cir_el.shape
    b = rows
    contpr = 0
    contop = 0
    contdc = 0
    conttr = 0
    for line in funtzioa:
        if ".TR" in np.char.upper(line):
            hasiera = float(line[1])
            bukaera = float(line[2])
            h = float(line[3])
        if ".PR" in np.char.upper(line):
            contpr += 1
            """1. praktikako gauza bera egin"""
            [cir_el, cir_nd, cir_val, cir_ctr, funtzioa] = zl1.cir_parser(
                filename
            )
            el_num = len(cir_el)
            [cir_el, cir_nd, cir_val, cir_ctr] = zl1.reshape(
                cir_el, cir_nd, cir_val, cir_ctr
            )
            rows_el, columns_el = cir_el.shape
            nodes = np.unique(cir_nd)
            b = rows_el
            n = len(np.unique(cir_nd))
            rows_ctr, columns_ctr = cir_ctr.shape
            zl1.print_cir_info(cir_el, cir_nd, b, n, nodes, el_num)
            incidence_matrix = zl1.incidence(cir_nd, nodes, n, b)
            print("\nIncidence Matrix: ")
            print(incidence_matrix)
    epsilon = 10 ** (-5)
    [j_ezlin, ezlin] = zl1.ez_linealak(cir_el)
    for line in funtzioa:
        if contpr > 1 or contop > 1 or contdc > 1 or conttr > 1:
            sys.exit("bakoitzeko analisi bakarra egin daiteke")
        elif ".OP" in np.char.upper(line):
            [cir_el, cir_nd, cir_val, cir_ctr, funtzioa] = zl1.cir_parser(
                filename
            )
            [cir_el, cir_nd, cir_val, cir_ctr] = zl1.reshape(
                cir_el, cir_nd, cir_val, cir_ctr
            )
            contop += 1
            [sol, b, n, g, V] = zl3.Ebatzi_denak(filename, cir_val, -1, 0)
            zl2.print_solution(sol, b, n)
        elif ".DC" in np.char.upper(line):
            contdc += 1
            hasiera = float(line[1])
            bukaera = float(line[2])
            pausua = float(line[3])
            sorgailua = line[4]
            [cir_el, cir_nd, cir_val, cir_ctr, funtzioa] = zl1.cir_parser(
                filename
            )
            [cir_el, cir_nd, cir_val, cir_ctr] = zl1.reshape(
                cir_el, cir_nd, cir_val, cir_ctr
            )
            j = 0
            i = 0
            for lerro in cir_el:
                if sorgailua.upper() in np.char.upper(lerro):
                    cir_val[i][0] = hasiera
                    ind = i
                i += 1
            while hasiera < bukaera:
                zenbaki = "{:.9f}".format(float(hasiera))
                lista = [zenbaki]
                [sol, b, n, g, V] = zl3.Ebatzi_denak(
                    filename, cir_val, hasiera, 0
                )
                for soluzio in sol:
                    lista.append(soluzio[0])
                if j == 0:
                    matrize = np.array(lista)
                else:
                    matrize = np.vstack([matrize, lista])
                hasiera = hasiera + float(pausua)
                cir_val[ind][0] = float(hasiera)
                j += 1
            filename = filename[:-4] + "_" + str(sorgailua) + ".dc"
            zl2.save_as_csv(b, n, filename, matrize, "v")
        elif ".TR" in np.char.upper(line):
            conttr += 1
            [cir_el, cir_nd, cir_val, cir_ctr, funtzioa] = zl1.cir_parser(
                filename
            )
            [cir_el, cir_nd, cir_val, cir_ctr] = zl1.reshape(
                cir_el, cir_nd, cir_val, cir_ctr
            )
            rows_el, columns_el = cir_el.shape
            nodes = np.unique(cir_nd)
            b = rows_el
            n = len(np.unique(cir_nd))
            j = 0
            indizeak_c = np.where(np.char.find(cir_el, "C") != -1)[0]
            indizeak_l = np.where(np.char.find(cir_el, "L") != -1)[0]
            if len(indizeak_c) > 0 or len(indizeak_l) > 0:
                while hasiera < bukaera:
                    i = 0
                    lista = [hasiera]
                    [sol, b, n, g, V] = zl3.Ebatzi_denak(
                        filename, cir_val, hasiera, h
                    )
                    for indize in indizeak_c:
                        Vc = sol[int(n - 1 + indize)]
                        cir_val[indize][1] = Vc
                    for indize in indizeak_l:
                        IL = sol[int(n - 1 + b + indize)]
                        cir_val[indize][1] = IL
                    for soluzio in sol:
                        lista.append(soluzio[0])
                    if j == 0:
                        matrize = np.array(lista)
                    else:
                        matrize = np.vstack([matrize, lista])
                    hasiera = hasiera + float(h)
                    j += 1
            else:
                while hasiera < bukaera:
                    i = 0
                    lista = [hasiera]
                    [sol, b, n, g, V] = zl3.Ebatzi_denak(
                        filename, cir_val, hasiera, h
                    )
                    for soluzio in sol:
                        lista.append(soluzio[0])
                    if j == 0:
                        matrize = np.array(lista)
                    else:
                        matrize = np.vstack([matrize, lista])
                    hasiera = hasiera + float(h)
                    j += 1
            filename = filename[:-4] + ".tr"
            zl2.save_as_csv(b, n, filename, matrize, "t")
