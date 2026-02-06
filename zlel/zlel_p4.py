#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

.. module:: zlel_p3.py
    :synopsis: Put yours

.. moduleauthor:: Put yours

"""

import time
import numpy as np
import sys

if __name__ == "zlel.zlel_p4":
    import zlel.zlel_p1 as zl1
    import zlel.zlel_p2 as zl2
    import zlel.zlel_p3 as zl3
else:
    import zlel_p1 as zl1
    import zlel_p2 as zl2
    import zlel_p3 as zl3

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "../cirs/all/1_zlel_opamp.cir"
    [cir_el, cir_nd, cir_val, cir_ctr, funtzioa] = zl1.cir_parser(filename)
    rows, cols = cir_el.shape
    b = rows
    for line in funtzioa:
        if ".TR" in np.char.upper(line):
            hasiera = float(line[1])
            bukaera = float(line[2])
            h = float(line[3])
    contpr = 0
    contop = 0
    contdc = 0
    conttr = 0
    for line in funtzioa:
        if contpr > 1 or contop > 1 or contdc > 1 or conttr > 1:
            sys.exit("bakoitzeko analisi bakarra egin daiteke")
        if ".PR" in np.char.upper(line):
            contpr += 1
            """1. praktikako gauza bera egin"""
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
            rows_ctr, columns_ctr = cir_ctr.shape
            el_num = rows_ctr
            zl1.print_cir_info(cir_el, cir_nd, b, n, nodes, el_num)
            incidence_matrix = zl1.incidence(cir_nd, nodes, n, b)
            print("\n incidence matrix: \n", incidence_matrix)
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
                    [M, N, Us] = zl1.MNUs(
                        cir_el, cir_val, cir_ctr, b, hasiera, 0, h, 0
                    )
                    A = zl1.incidence(cir_nd, nodes, n, b)
                    [T, sol] = zl2.Tableau(A, M, N, Us)
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
                    [M, N, Us] = zl1.MNUs(
                        cir_el, cir_val, cir_ctr, b, hasiera, 0, h, 0
                    )
                    A = zl1.incidence(cir_nd, nodes, n, b)
                    [T, sol] = zl2.Tableau(A, M, N, Us)
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
