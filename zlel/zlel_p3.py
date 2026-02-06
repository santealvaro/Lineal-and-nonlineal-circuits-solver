#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

.. module:: zlel_p3.py
    :synopsis: Put yours

.. moduleauthor:: Put yours

"""

import numpy as np
import sys
import math

if __name__ == "zlel.zlel_p3":
    import zlel.zlel_p1 as zl1
    import zlel.zlel_p2 as zl2
else:
    import zlel_p1 as zl1
    import zlel_p2 as zl2


def diode_NR(V, indizeak, epsilon, j, filename, cir_val_new, hasiera, h):
    """
    Calculates the g and the I of a diode for a NR discrete equivalent.
    Given,

        :math:`Id = I_0(e^{(\\frac{V_d}{nV_T})}-1)`

        The NR discrete equivalent will be,

        :math:`i_{j+1} + g v_{j+1} = I`

        where,

        :math:`g = -\\frac{I_0}{nV_T}e^{(\\frac{V_d}{nV_T})}`

        and

        :math:`I = I_0(e^{(\\frac{V_{dj}}{nV_T})}-1) + gV_{dj}`

    Args:
        **V**: List of voltages of non-linear elements. \n
        | **Indizeak**: List of index of the non-linear elements on
             the matrix cir_el.\n
        | **epsilon**: Accuracy for the NR function.\n
        | **j**: Index of the non-linear element that is getting evaluated.\n
        | **filename**: Name of the file of the .cir.\n
        | **cir_val_new**: The new cir_val, necessary for updating the
             cir_val matrix in .dc.\n
        | **hasiera**: argument for account time, necessary
            for dinamic circuits.\n
        | **h**: Time step.

    Returns:
        | **gd**: Conductance of the NR discrete equivalent for the diode.
        | **Id**: Current independent source of the NR discrete equivalent.
        | **V**: List of voltages of non-linear elements.
        | **Boolean**: used if there are more than one diode.

    """
    [cir_el, cir_nd, cir_val, cir_ctr, funtzioa] = zl1.cir_parser(filename)
    [cir_el, cir_nd, cir_val, cir_ctr] = zl1.reshape(
        cir_el, cir_nd, cir_val, cir_ctr
    )
    for line in funtzioa:
        if ".DC" in np.char.upper(line):
            cir_val = cir_val_new
        elif ".TR" in np.char.upper(line):
            cir_val = cir_val_new
    Nmax = 30
    i = 0
    Vd = V[j]
    n = cir_val[indizeak[j]][1]
    I0 = cir_val[indizeak[j]][0]
    while True:
        Vt = 8.6173324 * 10 ** (-5) * 300
        gd = -I0 / (n * Vt) * math.exp(V[j] / (n * Vt))
        Id = I0 * (math.exp(V[j] / (n * Vt)) - 1) + gd * V[j]
        rows_el, columns_el = cir_el.shape
        nodes = np.unique(cir_nd)
        b = rows_el
        ne = len(np.unique(cir_nd))
        [M, N, Us] = zl1.MNUs(cir_el, cir_val, cir_ctr, b, hasiera, V, h, 0)
        A = zl1.incidence(cir_nd, nodes, ne, b)
        [T, sol] = zl2.Tableau(A, M, N, Us)
        Vn = sol[int(ne - 1 + indizeak[j])]
        if abs(Vn - Vd) < epsilon:
            V[j] = Vn
            return gd, Id, V, True
        elif i == (Nmax - 1):
            V[j] = Vn
            print("iterazio kopuru maximoa gainditu da.")
            return gd, Id, V, False
        else:
            i = 0
            for x in V:
                V[i] = sol[int(ne - 1 + indizeak[i])]
                i += 1
            Vd = Vn
        i += 1
    return gd, Id, V, True


def Transistor_NR(V, indizeak, epsilon, j, filename, cir_val_new, hasiera, h):
    """
        Calculates the g-s,VBe and VBc of a transistor for a NR discrete
         equivalent
        Given,
            :math:`g_{11} = -I_{es} / V_t * e^{V_{Be} / V_t}`\n
            :math:`g_{22} = -I_{cs} / V_t * e^{V_{Bc} / V_t}`\n
            :math:`g_{12} = -a_r * g_{22}`\n
            :math:`g_{21} = -a_f * g_{11}`\n
            :math:`a_f = Bf / (1 + Bf)`\n
            :math:`a_r = Ies * af / Ics`\n
            :math:`I_e = g_{11} * V_{Be}+ g_{12} * V_{Bc}+ I_{es} *`
            :math:`(e^{V_{Be} / V_t} - 1)- a_r * I_{cs} *`
            :math:`(e^{V_{Bc} / V_t} - 1)`\n
            :math:`Ic = g21 * VBe+ g22 * VBc- af * Ies *`
            :math:`(e^{V_{Be} / V_t} - 1)+ Ics * (e^{V_{Bc} / V_t} - 1)`\n

        The NR discrete equivalent will be:
            :math:`i_e+g_{11}*V_{Be}+g_{12}*V_{Bc}=I_e`\n
            :math:`i_c+g_{21}*V_{Be}+g_{22}*V_{Bc}=I_c`
    Args:
        | **V**: List of voltages of non-linear elements.
        | **Indizeak**: List of index of the non-linear elements on the matrix
            cir_el.
        | **epsilon**: Accuracy for the NR function.
        | **j**: Index of the non-linear element that is getting evaluated.
        | **filename**: Name of the file of the .cir.
        | **cir_val_new**: The new cir_val, necessary for updating the cir_val
            matrix in .dc.
        | **hasiera**: argument for account time, necessary for
            dinamic circuits.
        | **h**: Time step.
    Returns:
        | **g**:Matrix formed by g=[[g11,g12],[g21,g22]],
            neccesary for solving circuits with transistors.
        | **V**: List of voltages of non-linear elements.
        | Boolean.

    """
    [cir_el, cir_nd, cir_val, cir_ctr, funtzioa] = zl1.cir_parser(filename)
    [cir_el, cir_nd, cir_val, cir_ctr] = zl1.reshape(
        cir_el, cir_nd, cir_val, cir_ctr
    )
    for line in funtzioa:
        if ".DC" in np.char.upper(line):
            cir_val = cir_val_new
        elif ".TR" in np.char.upper(line):
            cir_val = cir_val_new
    Nmax = 100
    kont = 0
    Vt = 8.6173324 * 10 ** (-5) * 300
    if "_BE" in cir_el[indizeak[j]][0].upper():
        VBe = V[j]
        VBc = V[j + 1]
    elif "_BC" in cir_el[indizeak[j]][0].upper():
        VBe = V[j - 1]
        VBc = V[j]
    Ies = cir_val[indizeak[j]][0]
    Ics = cir_val[indizeak[j]][1]
    Bf = cir_val[indizeak[j]][2]
    af = Bf / (1 + Bf)
    ar = Ies * af / Ics
    while True:
        g11 = -Ies / Vt * math.exp(VBe / Vt)
        g22 = -Ics / Vt * math.exp(VBc / Vt)
        g12 = -ar * g22
        g21 = -af * g11
        g = np.array([[g11, g12], [g21, g22]], dtype=float)
        rows_el, columns_el = cir_el.shape
        nodes = np.unique(cir_nd)
        b = rows_el
        ne = len(np.unique(cir_nd))
        [M, N, Us] = zl1.MNUs(cir_el, cir_val, cir_ctr, b, hasiera, V, h, g)
        A = zl1.incidence(cir_nd, nodes, ne, b)
        [T, sol] = zl2.Tableau(A, M, N, Us)
        if "_BE" in cir_el[indizeak[j]][0].upper():
            VBen = sol[int(ne - 1 + indizeak[j])]
            VBcn = sol[int(ne - 1 + indizeak[j + 1])]
            if abs(VBen - VBe) < epsilon:
                V[j] = VBen
                return g, V, True
            elif kont == (Nmax - 1):
                V[j] = VBen
                print("iterazio kopuru maximoa gainditu da.")
                return g, V, False
            else:
                i = 0
                for x in V:
                    V[i] = sol[int(ne - 1 + indizeak[i])]
                    i += 1
                VBe = VBen
                VBc = VBcn
        elif "_BC" in cir_el[indizeak[j]][0].upper():
            VBen = sol[int(ne - 1 + indizeak[j] - 1)]
            VBcn = sol[int(ne - 1 + indizeak[j])]
            if abs(VBcn - VBc) < epsilon:
                V[j] = VBcn
                return g, V, True
            elif kont == (Nmax - 1):
                V[j] = VBcn
                print("iterazio kopuru maximoa gainditu da.")
                return g, V, False
            else:
                i = 0
                for x in V:
                    V[i] = sol[int(ne - 1 + indizeak[i])]
                    i += 1
                VBe = VBen
                VBc = VBcn
        kont += 1
    return g, V, True


def Ebatzi_denak(filename, cir_val_dc, hasiera, h):
    """
        Function that returns the solution of the Tableau equations receiving
        the filename of a .cir.
    Args:
        | **filename**: Name of the file of the .cir.
        | **cir_val_dc**: The new cir_val,
             necessary for updating the cir_val matrix in .dc.
        | **hasiera**: argument for account time
             necessary for dinamic circuits.
        | **h**: Time step.
    Returns:
        **Sol**:List of all Tableau equation solutions,in order e,...,v,...,i.
        \n
        | **b**: dimensions of cir_el.\n
        | **n**: cir_el length.\n
        | **g**: Matrix formed by g=[[g11,g12],[g21,g22]],
             neccesary for solving circuits with transistors.\n
        | **V**: List of voltages of non-linear elements.
    """
    [cir_el, cir_nd, cir_val, cir_ctr, funtzioa] = zl1.cir_parser(filename)
    [cir_el, cir_nd, cir_val, cir_ctr] = zl1.reshape(
        cir_el, cir_nd, cir_val, cir_ctr
    )
    indizeak_c = np.where(np.char.find(cir_el, "C") != -1)[0]
    indizeak_l = np.where(np.char.find(cir_el, "L") != -1)[0]
    for line in funtzioa:
        if ".DC" in np.char.upper(line):
            cir_val = cir_val_dc
        elif ".TR" in np.char.upper(line):
            if len(indizeak_c) > 0 or len(indizeak_l) > 0:
                cir_val = cir_val_dc
    rows, cols = cir_el.shape
    b = rows
    epsilon = 10 ** (-5)
    [j_ezlin, ezlin] = zl1.ez_linealak(cir_el)
    rows, cols = cir_el.shape
    b = rows
    rows_el, columns_el = cir_el.shape
    nodes = np.unique(cir_nd)
    n = len(np.unique(cir_nd))
    A = zl1.incidence(cir_nd, nodes, n, b)
    g = np.zeros((2, 2))
    V = np.full(j_ezlin, 0.6, dtype=float)
    if j_ezlin > 0:
        indizeakd = np.where(np.char.find(cir_el, "D_") != -1)[0]
        i = 0
        for indize in indizeakd:
            [gd, Id, V, boolean] = diode_NR(
                V, indizeakd, epsilon, i, filename, cir_val_dc, hasiera, h
            )
            i += 1
        i = 0
        indizeakq = np.where(np.char.find(cir_el, "Q_") != -1)[0]
        for indize in indizeakq:
            [g, V, boolean] = Transistor_NR(
                V, indizeakq, epsilon, i, filename, cir_val_dc, hasiera, h
            )
            i += 1

        [M, N, Us] = zl1.MNUs(cir_el, cir_val, cir_ctr, b, hasiera, V, h, g)
        [T, sol] = zl2.Tableau(A, M, N, Us)
    else:
        [M, N, Us] = zl1.MNUs(cir_el, cir_val, cir_ctr, b, hasiera, V, h, g)
        [T, sol] = zl2.Tableau(A, M, N, Us)

    return sol, b, n, g, V


"""
https://stackoverflow.com/questions/419163/what-does-if-name-main-do
"""
if __name__ == "__main__":
    #  start = time.perf_counter()
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "../cirs/all/2_zlel_Q.cir"
        """Parse the circuit"""

#    print ("Elapsed time: ")
#    print(end - start) # Time in seconds
