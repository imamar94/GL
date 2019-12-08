import numpy as np
import scipy

class GraphLaplacian:
    def __init__(self, X0, Y0, X1, sigm=0.1):
        def weight_func(xi, xj, sigm):
            xixj = xi - xj
            xixj2 = np.dot(xixj, xixj)
            w = np.float16(np.exp(-(xixj2)/(sigm**2)))
            w = w if w > 1E-3 else 0
            return w
        
        self.__X0 = X0
        self.__Y0 = Y0
        self.__X1 = X1
        self.__solved = False
        self.__t = 0.5
        self.__wf = weight_func
        self.__sigm = sigm
        self.Y = list()
        self.U = list()
        self.X = list()
        
    @property
    def threshold(self):
        return self.__t
    
    @threshold.setter
    def threshold(self, v):
        self.__t = v
        if self.__solved:
            self.Y = np.array([1 if u>self.__t else 0 for u in list(self.U)])
        
    @property    
    def weight_function(self):
        return self.__wf
    
    @weight_function.setter
    def weight_function(self, f):
        self.__wf = f
        
    def create_linear_system(self):
        sigma = self.__sigm
        ## VARIABLE HELPER
        X = np.array(self.__X0 + self.__X1)
        self.X = X
        weight_func = self.__wf
        Nl = len(self.__X0)
        N = len(X)

        W = []
        for i, x in enumerate(X):
            w = [weight_func(X[i], X[j], sigma) if i != j else 0 for j in range(len(X))]
            W.append(w)
        W = np.array(W)

        A = []
        for i in range(Nl, N):
            a = [(-np.sum(W[i]) if i == j else W[i][j]) for j in range(Nl, N)]
            A.append(a)
        A = np.array(A)

        ## Create b
        b = np.array([-np.sum([w * self.__Y0[j] if j < Nl else 0 for j, w in enumerate(W[i])]) for i in range(Nl, N)])
        
        return A, b

    def solve(self, method="cg", ret=False):
        A, b = self.create_linear_system()
        if method == "cg":
            U = scipy.sparse.linalg.cg(A, b)[0]
        elif method == "npsolver":
            U = np.linalg.solve(A, b)
        else:
            raise(Exception("Unknown Method"))

        U = np.array(list(self.__Y0) + list(U))
        self.U = U
        Y = np.array([1 if u > self.__t else 0 for u in U])
        self.Y = Y
        self.__solved = True
        if ret:
            return U, Y


class MBOBinary:
    def __init__(self, X0, Y0, X1, sigm=0.5, dT=0.05, Nd=5):
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

        ## Initial Guest
        X = np.array(self.__X0 + self.__X1)
        Nl = len(self.__X0)
        Nu = len(self.__X1)
        N = len(X)
        U = np.ones(Nu)*0.5

        W = []
        for i, x in enumerate(X):
            w = [weight_func(X[i], X[j], sigm) if i!=j else 0 for j in range(len(X))]
            W.append(w)
        W = np.array(W)

        ## START MBO Iteration
        Unew = U
        for k in range(1000):
            for nd in range(self.__Nd):
                Uold = Unew
                A = []
                for i in range(Nl, N):
                    a = [(-np.sum(W[i])*dt - 1 if i==j else W[i][j]*dt) for j in range(Nl, N)]
                    A.append(a)

                A = np.array(A)
                b = np.array([-np.sum([w * self.__Y0[j] * dt if j < Nl else 0 for j, w in enumerate(W[i])])
                              - Unew[i-Nl]
                              for i in range(Nl, N)])

                Unew = scipy.sparse.linalg.cg(A, b)[0]
                error = np.dot(Unew - Uold, Unew - Uold) / np.dot(Unew, Unew)
                if error < 1e-5:
                    break

            ## Thresholding
            Unew = np.array([1 if u > 0.5 else 0 for u in Unew])
            error = np.dot(Unew - U, Unew - U) / np.dot(Unew, Unew)
            print("Interation",k,"error",error)
            if error < 1e-4:
                break
            U = Unew

        Unew = np.array(list(self.__Y0) + list(Unew))
        self.Y = Unew
        return Unew
