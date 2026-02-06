#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. module:: zlel_main.py
    :synopsis:

.. moduleauthor:: YOUR NAME AND E-MAIL


"""

import numpy as np
import sys


if __name__ == "zlel.zlel_p2":
    import zlel.zlel_p1 as zl1
else:
    import zlel_p1 as zl1


def print_solution(sol, b, n):
    """
    This function prints the solution with format.

    Args:
        | **sol**: np array with the solution of the Tableau equations
        | (e_1,..,e_n-1,v_1,..,v_b,i_1,..i_b)
        | **b**: # of branches
        | **n**: # of nodes

    """

    # The instructor solution needs to be a numpy array of numpy arrays of
    # float. If it is not, convert it to this format.
    if sol.dtype == np.float64:
        np.set_printoptions(sign=" ")  # Only from numpy 1.14
        tmp = np.zeros([np.size(sol), 1], dtype=float)
        for ind in range(np.size(sol)):
            tmp[ind] = np.array(sol[ind])
        sol = tmp
    print("\n========== Nodes voltage to reference ========")
    for i in range(1, n):
        print("e" + str(i) + " = ", "[{:10.9f}]".format(sol[i - 1][0]))
    print("\n========== Branches voltage difference ========")
    for i in range(1, b + 1):
        print("v" + str(i) + " = ", "[{:10.9f}]".format(sol[i + n - 2][0]))
    print("\n=============== Branches currents ==============")
    for i in range(1, b + 1):
        print("i" + str(i) + " = ", "[{:10.9f}]".format(sol[i + b + n - 2][0]))

    print("\n================= End solution =================\n")


def build_csv_header(tvi, b, n):
    """
    This function builds the csv header for the files.\n
    First column will be v or i if .dc analysis or t if .tr and it will
    be given by argument tvi.
    The header will be this form,
    **t/v/i,e_1,..,e_n-1,v_1,..,v_b,i_1,..i_b**

    Args:
        | **tvi**: "v" or "i" if .dc analysis or "t" if .tran
        | **b**: # of branches
        | **n**: # of nodes

    Returns:
        **header**: The header in csv format as string
    """
    header = tvi
    for i in range(1, n):
        header += ",e" + str(i)
    for i in range(1, b + 1):
        header += ",v" + str(i)
    for i in range(1, b + 1):
        header += ",i" + str(i)
    return header


def save_as_csv(b, n, filename, matrize, tvi):
    """
    This function gnerates a csv file with the name filename. First
    it will save a header and then, it loops and save a line in
    csv format into the file.

    Args:
        | **b**: # of branches
        | **n**: # of nodes
        | **filename**: string with the filename (incluiding the path)
    """
    # Sup .tr
    header = build_csv_header(tvi, b, n)
    with open(filename, "w") as file:
        print(header, file=file)
        # Get the indices of the elements corresponding to the sources.
        # The freq parameter cannot be 0 this is why we choose cir_tr[0].
        for line in matrize:
            i = 1
            string = ""
            for i in range(len(line)):
                if i == 0:
                    string = string + "{:.9f}".format(float(line[i]))
                else:
                    string = string + ",{:.9f}".format(float(line[i]))
            print(string, file=file)


def plot_from_cvs(filename, x, y, title):
    """
    This function plots the values corresponding to the x string of the
    file filename in the x-axis and the ones corresponding to the y
    string in the y-axis.
    The x and y strings must mach with some value of the header in the
    csv file filename.

    Args:
        | **filename**: string with the name of the file (including the path).
        | **x**: string with some value of the header of the file.
        | **y**: string with some value of the header of the file.

    """
    data = np.genfromtxt(
        filename, delimiter=",", skip_header=0, skip_footer=1, names=True
    )

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(data[x], data[y], color="r", label=title)
    ax1.set_xlabel(x)
    ax1.set_ylabel(y)
    plt.show()


def Tableau(a, M, N, Us):
    """
    This function evaluates the Tableau equations,
    using the M,N and Us matrices and the A incidence matrix.\n
    Args:
        | **M**: Voltage matrix.
        | **N**: Current matrix.
        | **Us**: Vector of non V controlled elements.
    Returns:
        | **T**: Tableau matrix, formed by all Tableau equations,
            in order e,...,v,...,i.
        **Sol**: List of all Tableau equation solutions, in order e,...,v,...,i
    """
    A = np.array(a[1:])
    b1 = np.shape(A)[0]
    b2 = np.shape(A)[1]
    T = np.zeros((b1 + 2 * b2, b1 + 2 * b2), dtype=float)
    u = np.zeros((b1 + 2 * b2, 1), dtype=float)
    i = 0
    while i < (b1):
        j = 0
        while j < (b1 + 2 * b2):
            if (b1 + b2) <= j:
                T[i][j] = A[i][j - b1 - b2]
            j += 1
        i += 1
    while i < (b1 + b2):
        j = 0
        while j < (b1):
            T[i][j] = -np.transpose(A)[i - b1][j]
            j += 1
        while j < (b1 + b2):
            T[i][j] = np.eye(b2)[i - b1][j - b1]
            j += 1
        i += 1
    while i < (b1 + 2 * b2):
        j = b1
        while (b1) <= j < (b1 + b2):
            T[i][j] = M[i - b1 - b2][j - b1]
            j += 1
        while (b1 + b2) <= j < (b1 + 2 * b2):
            T[i][j] = N[i - b1 - b2][j - b1 - b2]
            j += 1
        i += 1
    i = 0
    while i < (b1 + 2 * b2):
        while (b1 + b2) <= i < (b1 + 2 * b2):
            u[i] = Us[i - b1 - b2]
            i += 1
        i += 1
    if np.linalg.det(T) == 0:
        sys.exit("Error solving Tableau equations, check if det(T) != 0.")
    else:
        sol = np.linalg.solve(T, u)
    return T, sol


"""
https://stackoverflow.com/questions/419163/what-does-if-name-main-do
https://stackoverflow.com/questions/19747371/
python-exit-commands-why-so-many-and-when-should-each-be-used
"""

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "../cirs/all/1_zlel_anpli.cir"
    [cir_el, cir_nd, cir_val, cir_ctr, funtzioa] = zl1.cir_parser(filename)
    rows, cols = cir_el.shape
    b = rows
    contpr = 0
    contop = 0
    contdc = 0
    conttr = 0
    for line in funtzioa:
        if contpr > 1 or contop > 1 or contdc > 1 or conttr > 1:
            sys.exit("bakoitzeko analisi bakarra egin daiteke")
        if ".PR" in np.char.upper(line):
            contpr += 1
            print(True)
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
        elif ".OP" in np.char.upper(line):
            contop += 1
            """M, N eta U lortu eta sistema askatu"""
            [cir_el, cir_nd, cir_val, cir_ctr, funtzioa] = zl1.cir_parser(
                filename
            )
            [cir_el, cir_nd, cir_val, cir_ctr] = zl1.reshape(
                cir_el, cir_nd, cir_val, cir_ctr
            )
            rows, cols = cir_el.shape
            b = rows
            [M, N, Us] = zl1.MNUs(cir_el, cir_val, cir_ctr, b, 0, 0)
            nodes = np.unique(cir_nd)
            n = len(np.unique(cir_nd))
            A = zl1.incidence(cir_nd, nodes, n, b)
            [T, sol] = Tableau(A, M, N, Us)
            print_solution(sol, b, n)
        elif ".DC" in np.char.upper(line):
            contdc += 1
            hasiera = int(line[1])
            bukaera = int(line[2])
            pausua = int(line[3])
            sorgailua = line[4]
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
            i = 0
            for lerro in cir_el:
                if sorgailua.upper() in np.char.upper(lerro):
                    cir_val[i][0] = hasiera
                    i += 1
            while hasiera < bukaera:
                i = 0
                zenbaki = "{:.9f}".format(float(hasiera))
                lista = [zenbaki]
                [M, N, Us] = zl1.MNUs(cir_el, cir_val, cir_ctr, b, 0, 0, 0)
                A = zl1.incidence(cir_nd, nodes, n, b)
                [T, sol] = Tableau(A, M, N, Us)
                for soluzio in sol:
                    lista.append(soluzio[0])
                if j == 0:
                    matrize = np.array(lista)
                else:
                    matrize = np.vstack([matrize, lista])
                for lerro in cir_el:
                    if sorgailua.upper() in np.char.upper(lerro):
                        temp = int(cir_val[i][0])
                        cir_val[i][0] = int(temp) + int(pausua)
                    i += 1
                hasiera = hasiera + int(pausua)
                j += 1
            filename = filename[:-4] + "_" + str(sorgailua) + ".dc"
            save_as_csv(b, n, filename, matrize, "v")
        elif ".TR" in np.char.upper(line):
            conttr += 1
            hasiera = float(line[1])
            bukaera = float(line[2])
            pausua = float(line[3])
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
            while hasiera < bukaera:
                i = 0
                lista = [hasiera]
                [M, N, Us] = zl1.MNUs(
                    cir_el, cir_val, cir_ctr, b, hasiera, 0, 0, 0
                )
                A = zl1.incidence(cir_nd, nodes, n, b)
                [T, sol] = Tableau(A, M, N, Us)
                for soluzio in sol:
                    lista.append(soluzio[0])
                if j == 0:
                    matrize = np.array(lista)
                else:
                    matrize = np.vstack([matrize, lista])
                hasiera = hasiera + float(pausua)
                j += 1
            filename = filename[:-4] + ".tr"
            save_as_csv(b, n, filename, matrize, "t")
