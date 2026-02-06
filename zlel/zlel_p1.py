#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. module:: zlel_maiy
    :synopsis:

.. moduleauthor:: YOUR NAME AND E-MAIL


"""

import numpy as np
import sys
import math


def cir_parser(filename):
    """
        This function takes a .cir test circuit and parse it into
        4 matices.
        If the file has not the proper dimensions it warns and exit.
        This function also ensures that the cir does not contain any of the
        following errors:\\

        |   There is a **reference node** defined.\n
        |   There are **no floating nodes**.\n
        |   There are **no parallel V sources**.\n
        |   There are **no serial I sources**.
    Args:
        **filename**: string with the name of the file
    Returns:
        | **cir_el**: np array of strings with the elements to parse. size(1,b)
        | **cir_nd**: np array with the nodes to the circuit. size(b,4)
        | **cir_val**: np array with the values of the elements. size(b,3)
        | **cir_ctrl**: np array of strings with the element which branch
        | controls the controlled sources. size(1,b)

    Rises:
        SystemExit
    """
    try:
        cir = np.array(np.loadtxt(filename, dtype=str))
    except ValueError:
        sys.exit("File corrupted: .cir size is incorrect.")
    i = 0
    funtzioa = np.array([0, 0, 0, 0, 0])
    for line in cir:
        if "." in line[0][0]:
            lerro = np.array([line[0], line[5], line[6], line[7], line[8]])
            funtzioa = np.vstack([funtzioa, lerro])
            i += 1
    cir_el = np.array(cir[:-i, 0:1], dtype=str)
    # THIS FUNCTION IS NOT COMPLETE
    cir_nd = np.array(cir[:-i, 1:5], dtype=int)
    cir_val = np.array(cir[:-i, 5:8], dtype=float)
    cir_ctrl = np.array(cir[:-i, 8:], dtype=str)
    reference_node = cir_nd[:, 0:2]
    b = reference_node[reference_node < 1]
    if len(b) < 1:
        sys.exit('Reference node "0" is not defined in the circuit.')
    hiztegi = {}
    i = 0
    for lerro in cir_nd:
        j = 0
        for elementu in lerro:
            if "Q" in np.char.upper(cir_el[i][0][0]) or "A" in np.char.upper(
                cir_el[i][0][0]
            ):
                if elementu in hiztegi:
                    hiztegi[elementu] += 1
                else:
                    hiztegi[elementu] = 1
                if j == 2:
                    break
            else:
                if elementu in hiztegi:
                    hiztegi[elementu] += 1
                else:
                    hiztegi[elementu] = 1
                if j == 1:
                    break
            j += 1
        i += 1
    for giltza, balioak in hiztegi.items():
        if balioak == 1:
            sys.exit("Node " + str(giltza) + " is floating.")
    for line in cir:
        k = 0
        for line2 in cir:
            if line[0] != line2[0]:
                if (
                    "V" in line[0][0].upper()
                    or "E" in line[0][0].upper()
                    or "H" in line[0][0].upper()
                    or "B" in line[0][0].upper()
                ) and (
                    "V" in line2[0][0].upper()
                    or "E" in line2[0][0].upper()
                    or "H" in line2[0][0].upper()
                    or "B" in line2[0][0].upper()
                ):
                    if (
                        (line[1] == line2[1] and line[2] == line2[2])
                        or (line[1] == line2[2] and line[2] == line2[1])
                    ) and (
                        (line[5] != line2[5])
                        or (line[1] == line2[2] and line[2] == line2[1])
                    ):
                        sys.exit(
                            "Parallel V sources at branches "
                            + str(
                                min(
                                    np.where(
                                        np.char.find(cir_el, line[0]) != -1
                                    )[0][0],
                                    np.where(
                                        np.char.find(cir_el, line2[0]) != -1
                                    )[0][0],
                                )
                            )
                            + " and "
                            + str(
                                max(
                                    np.where(
                                        np.char.find(cir_el, line[0]) != -1
                                    )[0][0],
                                    np.where(
                                        np.char.find(cir_el, line2[0]) != -1
                                    )[0][0],
                                )
                            )
                            + "."
                        )
                if (
                    "I" in line[0][0].upper()
                    or "G" in line[0][0].upper()
                    or "F" in line[0][0].upper()
                    or "Y" in line[0][0].upper()
                ) and (
                    "I" in line2[0][0].upper()
                    or "G" in line2[0][0].upper()
                    or "F" in line2[0][0].upper()
                    or "Y" in line2[0][0].upper()
                ):
                    if (
                        (line[1] == line2[1])
                        or (line[1] == line2[2])
                        or (line[2] == line2[1])
                        or (line[2] == line2[2])
                    ):
                        if (line[1] == line2[1]) or (line[1] == line2[2]):
                            a = line[1]
                        if (line[2] == line2[1]) or (line[2] == line2[2]):
                            a = line[2]
                        x = 0
                        for line3 in cir:
                            if not (
                                "I" in line3[0][0].upper()
                                or "G" in line3[0][0].upper()
                            ) and (
                                (
                                    (
                                        line3[1] == line2[1]
                                        and line3[2] == line2[2]
                                    )
                                    or (
                                        line3[1] == line2[2]
                                        and line3[2] == line2[1]
                                    )
                                )
                                or (
                                    (
                                        line3[1] == line[1]
                                        and line3[2] == line[2]
                                    )
                                    or (
                                        line3[1] == line[2]
                                        and line3[2] == line[1]
                                    )
                                )
                            ):

                                x = 1
                        if x == 0:
                            for line3 in cir:
                                if (
                                    "I" in line3[0][0].upper()
                                    or "G" in line3[0][0].upper()
                                ) and ((line3[1] == a)):
                                    k -= int(line3[5])
                                if (
                                    "I" in line3[0][0].upper()
                                    or "G" in line3[0][0].upper()
                                ) and (line3[2] == a):
                                    k += int(line3[5])
                        if k != 0:
                            sys.exit(
                                "I sources in series at node " + str(a) + "."
                            )
    b = len(cir_el)

    return cir_el, cir_nd, cir_val, cir_ctrl, funtzioa


def reshape(cir_el, cir_nd, cir_val, cir_ctr):
    """
    Function that will resize the two matrix by taking into account
    if there are any transistros or OPAMP \n
    Args:
        **cir_el**: matrix got when parsing the circuit (circuit elements).\n
        **cir_nd**: matrix got when parsing the circuit (nodes of each element)
    Returns:
        reshaped cir_el and cir_nd matrixes
    """
    i = 0
    for nodoak in cir_nd:
        elementua = cir_el[i][0]
        if "Q_" in elementua.upper():
            if i == 0:
                bc = cir_el[i][0] + "_be"
                be = cir_el[i][0] + "_bc"
                temp1 = [bc]
                temp2 = [be]
                temp3 = [nodoak[1], nodoak[2]]
                temp4 = [nodoak[1], nodoak[0]]
                new_elements = np.array([temp1, temp2])
                new_nodes = np.array([temp3, temp4])
                new_values = np.array([cir_val[i], cir_val[i]])
                new_ctr = np.array([cir_ctr[i], cir_ctr[i]])
            else:
                bc = cir_el[i][0] + "_be"
                be = cir_el[i][0] + "_bc"
                temp1 = [bc]
                temp2 = [be]
                temp3 = [nodoak[1], nodoak[2]]
                temp4 = [nodoak[1], nodoak[0]]
                new_elements = np.vstack([new_elements, temp1, temp2])
                new_nodes = np.vstack([new_nodes, temp3, temp4])
                new_values = np.vstack([new_values, cir_val[i], cir_val[i]])
                new_ctr = np.vstack([new_ctr, cir_ctr[i], cir_ctr[i]])
        elif "A_" in elementua.upper():
            if i == 0:
                In = str(cir_el[i][0]) + "_in"
                out = str(cir_el[i][0]) + "_ou"
                temp1 = [In]
                temp2 = [out]
                temp3 = [nodoak[0], nodoak[1]]
                temp4 = [nodoak[2], nodoak[3]]
                new_elements = np.array([temp1, temp2])
                new_nodes = np.array([temp3, temp4])
                new_values = np.array([cir_val[i], cir_val[i]])
                new_ctr = np.array([cir_ctr[i], cir_ctr[i]])
            else:
                In = str(cir_el[i][0]) + "_in"
                out = str(cir_el[i][0]) + "_ou"
                temp1 = [In]
                temp2 = [out]
                temp3 = [nodoak[0], nodoak[1]]
                temp4 = [nodoak[2], nodoak[3]]
                new_elements = np.vstack([new_elements, temp1, temp2])
                new_nodes = np.vstack([new_nodes, temp3, temp4])
                new_values = np.vstack([new_values, cir_val[i], cir_val[i]])
                new_ctr = np.vstack([new_ctr, cir_ctr[i], cir_ctr[i]])
        else:
            if i == 0:
                temp1 = [cir_el[i][0]]
                temp2 = [nodoak[0], nodoak[1]]
                new_elements = np.array(temp1)
                new_nodes = np.array(temp2)
                new_values = np.array(cir_val[i])
                new_ctr = np.array(cir_ctr[i])
            else:
                temp1 = [cir_el[i][0]]
                temp2 = [nodoak[0], nodoak[1]]
                new_elements = np.vstack([new_elements, temp1])
                new_nodes = np.vstack([new_nodes, temp2])
                new_values = np.vstack([new_values, cir_val[i]])
                new_ctr = np.vstack([new_ctr, cir_ctr[i]])
        i = i + 1
    cir_el = new_elements
    cir_nd = new_nodes
    cir_val = new_values
    cir_ctr = new_ctr
    return cir_el, cir_nd, cir_val, cir_ctr


def print_cir_info(cir_el, cir_nd, b, n, nodes, el_num):
    """
    Prints the info of the circuit:
        |     1.- Elements info
        |     2.- Node info
        |     3.- Branch info
        |     4.- Variable info
    Args:
        | **cir_el**: reshaped cir_el
        | **cir_nd**: reshaped cir_nd. Now it will be a(b,2) matrix
        | **b**: number of branches
        | **n**: number of nodes
        | **nodes**: an array with the circuit nodes sorted
        | **el_num**:  the number of elements
    """
    # Element info
    print(str(el_num) + " Elements")
    # Node info
    print(str(n) + " Different nodes: " + str(nodes))
    # Branch info
    print("\n" + str(b) + " Branches: ")

    for i in range(1, b + 1):
        indent = 12  # Number of blanks for indent
        string = (
            "\t"
            + str(i)
            + ". branch:\t"
            + str(cir_el[i - 1][0])
            + "i".rjust(indent - len(cir_el[i - 1][0]))
            + str(i)
            + "v".rjust(indent - len(str(i)))
            + str(i)
            + " = e"
            + str(cir_nd[i - 1, 0])
            + " - e"
            + str(cir_nd[i - 1, 1])
        )
        print(string)

    # Variable info
    print("\n" + str(2 * b + (n - 1)) + " variables: ")
    # print all the nodes but the first(0 because is sorted)
    for i in nodes[1:]:
        print("e" + str(i) + ", ", end="", flush=True)
    for i in range(b):
        print("i" + str(i + 1) + ", ", end="", flush=True)
    # print all the branches but the last to close it properly
    # It works because the minuimum amount of branches in a circuit must be 2.
    for i in range(b - 1):
        print("v" + str(i + 1) + ", ", end="", flush=True)
    print("v" + str(b))

    # IT IS RECOMMENDED TO USE THIS FUNCTION WITH NO MODIFICATION.


def incidence(cir_nd, nodes, n, b):
    """Creates an incidence matrix based on the reshaped cir_nd. \n
    Args:
        **cir_nd**: The reshaped cir_nd.\n
        **nodes**: An array containing all of the different nodes.\n
        **n**: The number of nodes.\n
        **b**: The number of branches.
    Returns:
        **matrix**: The incidence matrix.
    """
    matrix = np.zeros((n, b), dtype=int)
    i = 0
    for line in cir_nd:
        pos1 = np.where(nodes == int(line[0]))[0]
        pos2 = np.where(nodes == int(line[1]))[0]
        matrix[int(pos1)][i] = 1
        matrix[int(pos2)][i] = -1
        i = i + 1
    return matrix


def ez_linealak(cir_el):
    """function that cheks if there are any diodes or transistors.\n
    Args:
        **cir_el**: matrix got when parsing the circuit (circuit elements).
    Returns:
        **j**: The number of non lineal elements.\n
        **ezlin**: matrix with the non lineal elements.
    """
    ezlin = np.array([])
    i = 0
    j = 0
    for line in cir_el:
        if "D" in line[0].upper():
            ezlin = np.append(ezlin, i)
            j += 1
        elif "Q" in line[0].upper():
            ezlin = np.append(ezlin, i)
            j += 1
        i += 1
    return j, ezlin


def MNUs(cir_el, cir_val, cir_ctr, b, denbora, V, h, g):
    """This function creates the M, N and Us matrices
    that are necessary to form the Tableau equations.\n
    Args:
        | **cir_el**:
            np array of strings with the elements to parse. size(1,b)
        | **cir_val**:
            np array with the values of the elements. size(b,3)
        | **cir_ctrl**:
            np array of strings with the element which branch
        | **b**: dimensions of cir_el.
        | **denbora**:
            argument for account time, necessary for dinamic circuits.
        | **V**: List of voltages of non-linear elements.
        | **h**: Time step.
        | **g**: Matrix formed by g
            =[[g11,g12],[g21,g22]],
            neccesary for solving circuits with transistors.
    Returns:
        | **M**: "Voltage matrix".
        | **N**: "Current matrix".
        | **Us**: Vector of non V controlled elements.
    """
    i = 0
    t = float(denbora)
    Vt = 8.6173324 * 10 ** (-5) * 300
    M = np.zeros((b, b), dtype=float)
    N = np.zeros((b, b), dtype=float)
    Us = np.zeros(b, dtype=float)
    j = 0
    for line in cir_el:
        if "R_" in line[0].upper():
            M[i][i] = 1
            N[i][i] = -cir_val[i, 0]
            Us[i] = 0
            i += 1
        elif "V_" in line[0].upper():
            M[i][i] = 1
            N[i][i] = 0
            Us[i] = cir_val[i, 0]
            i += 1
        elif "I_" in line[0].upper():
            M[i][i] = 0
            N[i][i] = 1
            Us[i] = cir_val[i, 0]
            i += 1
        elif "A_" in line[0].upper():
            if "IN" in line[0].upper():
                M[i][i] = 0
                N[i][i] = 1
                Us[i] = 0
                i += 1
            else:
                M[i][i - 1] = 1
                N[i][i] = 0
                Us[i] = 0
                i += 1
        elif "D_" in line[0].upper():
            Vd = V[j]
            M[i][i] = (
                -cir_val[i][0]
                / (cir_val[i][1] * Vt)
                * math.exp(Vd / (cir_val[i][1] * Vt))
            )
            N[i][i] = 1
            Us[i] = (
                cir_val[i][0] * (math.exp(Vd / (cir_val[i][1] * Vt)) - 1)
                + M[i][i] * Vd
            )
            i += 1
            j += 1
        elif "_BE" in line[0].upper():
            g11 = g[0][0]
            g12 = g[0][1]
            Ies = cir_val[i][0]
            Ics = cir_val[i][1]
            Bf = cir_val[i][2]
            af = Bf / (1 + Bf)
            ar = Ies * af / Ics
            VBe = V[j]
            VBc = V[j + 1]
            Ie = (
                g11 * VBe
                + g12 * VBc
                + Ies * (math.exp(VBe / Vt) - 1)
                - ar * Ics * (math.exp(VBc / Vt) - 1)
            )
            N[i][i] = 1
            M[i][i] = g11
            M[i][i + 1] = g12
            Us[i] = Ie
            i += 1
            j += 1
        elif "_BC" in line[0].upper():
            g21 = g[1][0]
            g22 = g[1][1]
            Ies = cir_val[i][0]
            Ics = cir_val[i][1]
            VBe = V[j - 1]
            VBc = V[j]
            Bf = cir_val[i][2]
            af = Bf / (1 + Bf)
            Ic = (
                g21 * VBe
                + g22 * VBc
                - af * Ies * (math.exp(VBe / Vt) - 1)
                + Ics * (math.exp(VBc / Vt) - 1)
            )
            N[i][i] = 1
            M[i][i] = g22
            M[i][i - 1] = g21
            Us[i] = Ic
            i += 1
            j += 1
        elif "C_" in line[0].upper():
            M[i][i] = 1
            N[i][i] = -h / cir_val[i][0]
            Us[i] = cir_val[i][1]
            i += 1
        elif "L_" in line[0].upper():
            M[i][i] = -h / cir_val[i][0]
            N[i][i] = 1
            Us[i] = cir_val[i][1]
            i += 1
        elif "E_" in line[0].upper():
            """Tentsioen bidez kontrolatutako tentsio iturria"""
            k = 0
            for x in cir_el:
                if x[0].upper() in np.char.upper(cir_ctr[i]):
                    M[i][k] = -cir_val[i, 0]
                k += 1
            M[i][i] = 1
            N[i][i] = 0
            Us[i] = 0
            i += 1
        elif "G_" in line[0].upper():
            """Tentsioen bidez kontrolatutako korronte iturria"""
            k = 0
            for x in cir_el:
                if x[0].upper() in np.char.upper(cir_ctr[i]):
                    M[i][k] = -cir_val[i, 0]
                k += 1
            M[i][i] = 0
            N[i][i] = 1
            Us[i] = 0
            i += 1
        elif "H_" in line[0].upper():
            """Korronteren bidez kontrolatutako tentsio iturria"""
            k = 0
            for x in cir_el:
                if x[0].upper() in np.char.upper(cir_ctr[i]):

                    N[i][k] = -cir_val[i, 0]
                k += 1
            M[i][i] = 1
            N[i][i] = 0
            Us[i] = 0
            i += 1
        elif "F_" in line[0].upper():
            """Korronteren bidez kontrolatutako korronte iturria"""
            k = 0
            for x in cir_el:
                if x[0].upper() in np.char.upper(cir_ctr[i]):

                    N[i][k] = -cir_val[i, 0]
                k += 1
            M[i][i] = 0
            N[i][i] = 1
            Us[i] = 0
            i += 1
        elif "B_" in line[0].upper():
            M[i][i] = 1
            N[i][i] = 0
            if t == -1:
                Us[i] = cir_val[i, 0]
            else:
                Us[i] = cir_val[i, 0] * math.sin(
                    2 * math.pi * cir_val[i, 1] * t
                    + math.pi * cir_val[i, 2] / 180
                )
            i += 1
        elif "Y_" in line[0].upper():
            M[i][i] = 0
            N[i][i] = 1
            if t == -1:
                Us[i] = cir_val[i, 0]
            else:
                Us[i] = cir_val[i, 0] * math.sin(
                    2 * math.pi * cir_val[i, 1] * t
                    + math.pi * cir_val[i, 2] / 180
                )
            i += 1
    return M, N, Us


"""
https://stackoverflow.com/questions/419163/what-does-if-name-main-do
https://stackoverflow.com/questions/19747371/
python-exit-commands-why-so-many-and-when-should-each-be-used
"""
if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "../cirs/all/1_zlel_opamp.cir"
    """Parse the circuit"""
    [cir_el, cir_nd, cir_val, cir_ctr, funtzioa] = cir_parser(filename)
    [cir_el, cir_nd, cir_val, cir_ctr] = reshape(
        cir_el, cir_nd, cir_val, cir_ctr
    )
    """print info"""
    rows_el, columns_el = cir_el.shape
    nodes = np.unique(cir_nd)
    b = rows_el
    n = len(np.unique(cir_nd))
    rows_ctr, columns_ctr = cir_ctr.shape
    el_num = rows_ctr
    print_cir_info(cir_el, cir_nd, b, n, nodes, el_num)
    """intzidentzia matrizea"""
    incidence_matrix = incidence(cir_nd, nodes, n, b)
    print("\nincidence matrix:")
    print(incidence_matrix)

#    THIS FUNCTION IS NOT COMPLETE
