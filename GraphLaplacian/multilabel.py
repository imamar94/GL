import numpy as np
from .binary import GraphLaplacian

import math

class GLMultiClass:
    def __init__(self, X0, Y0, X1, sigm=0.1):
        def weight_func(xi, xj, sigm):
            xixj = xi - xj
            xixj2 = np.dot(xixj, xixj)
            w = np.float16(np.exp(-(xixj2) / (sigm ** 2)))
            w = w if w > 1E-3 else 0
            return w

        self.__X0 = X0
        self.__Y0 = Y0
        self.__X1 = X1
        self.__solved = False
        self.__wf = weight_func
        self.__sigm = sigm
        self.Y = list()
        self.U = list()

    @property
    def weight_function(self):
        return self.__wf

    @weight_function.setter
    def weight_function(self, f):
        self.__wf = f

    def solve(self):
        Uk = list()
        for label in set(self.__Y0):
            Y0_ = [1 if y == label else 0 for y in self.__Y0]
            GL = GraphLaplacian(self.__X0, Y0_, self.__X1,
                                sigm=self.__sigm)
            GL.weight_function = self.__wf
            GL.solve(method="npsolver")
            Uk.append(GL.U)

        self.U = np.array(Uk).T
        self.Y = np.array([np.argmax(u) for u in np.array(Uk).T])



class MBOMultiClass:
    def __init__(self, X0, Y0, X1, sigm=0.5, dT=0.05, Nd=3):
        def weight_func(xi, xj, sigm):
            xixj = xi - xj
            xixj2 = np.dot(xixj, xixj)
            w = np.float16(np.exp(-(xixj2) / (sigm ** 2)))
            w = w if w > 1E-3 else 0
            return w

        self.__X0 = X0
        self.__Y0 = Y0
        self.__X1 = X1
        self.__solved = False
        self.__wf = weight_func
        self.__sigm = sigm
        self.__dt = dT/Nd
        self.__Nd = Nd
        self.Y = list()
        self.X = list()

    @property
    def weight_function(self):
        return self.__wf

    @weight_function.setter
    def weight_function(self, f):
        self.__wf = f

    def solve(self):
        weight_func = self.__wf
        sigm = self.__sigm
        dt = self.__dt

        ## Initial Guess
        X = np.array(self.__X0 + self.__X1)
        Nl = len(self.__X0)
        Nu = len(self.__X1)
        N = len(X)
        n_classes = max(self.__Y0) + 1
        temp = range(n_classes)
        Y0t = [[1 if x == y else 0 for x in temp] for y in self.__Y0]
        U = np.ones([Nu, n_classes])*0.5

        W = []
        for i, x in enumerate(X):
            w = [weight_func(X[i], X[j], sigm) if i != j else 0 for j in range(len(X))]
            W.append(w)
        W = np.array(W)

        ## START MBO Iteration
        Unew = U
        for k in range(1000):
            for nd in range(self.__Nd):
                Uold = Unew
                A = []
                for i in range(Nl, N):
                    a = [(-np.sum(W[i])*dt - 1 if i == j else W[i][j]*dt) for j in range(Nl, N)]
                    A.append(a)

                A = np.array(A)

                b = list()
                for c in range(n_classes):
                    b_class = np.array(
                        [-np.sum([w * Y0t[j][c] * dt if j < Nl else 0 for j, w in enumerate(W[i])])
                         - Unew[i-Nl][c]
                         for i in range(Nl, N)]
                    )
                    b.append(b_class)
                b = np.array(b).T

                Unew = np.dot(np.linalg.inv(A), b)
                error = np.sum((Unew - Uold) * (Unew - Uold)) / np.sum(Unew * Unew)
                if error < 1e-5:
                    break

            ## Thresholding
            Unew = np.array([[1 if np.argmax(u) == x else 0 for x in temp] for u in Unew])
            error = np.sum((Unew - U) * (Unew - U)) / np.sum(Unew * U)
            print(error)
            if error < 1e-4:
                break
            U = Unew

        self.Y = list(self.__Y0) + [np.argmax(u) for u in Unew]
