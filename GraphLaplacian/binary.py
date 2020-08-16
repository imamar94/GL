import numpy as np
import scipy
from scipy.stats import multivariate_normal
from datetime import datetime


class GraphLaplacian:
    def __init__(self, X0, Y0, X1, sigma=0.1, weight_matrix=None, w_cut=0.001):
        self._X0 = X0
        self._Y0 = Y0
        self._X1 = X1
        self._solved = False
        self._t = 0.5
        self._sigma = sigma
        self.Y = list()
        self.U = list()
        self.X = list()
        self.weight_matrix = weight_matrix
        self.w_cut = w_cut

    @property
    def threshold(self):
        return self._t

    @threshold.setter
    def threshold(self, v):
        self._t = v
        if self._solved:
            self.Y = np.array([1 if u > self._t else 0 for u in list(self.U)])

    def create_linear_system(self):
        sigma = self._sigma
        ## VARIABLE HELPER
        X = np.array(self._X0 + self._X1)
        self.X = X
        Nl = len(self._X0)
        N = len(X)
        n_feature = len(X[0])

        # try:
        #     _ = iter(cov)
        #     if cov.ndim == 1:
        #         # print("Sampe Sini")
        #         cov = np.diag(cov * cov)
        #     else:
        #         pass
        # except TypeError:
        #     cov = np.diag((cov * cov) * np.ones(n_feature))
        # W = []
        # tic2 = datetime.now()

        if isinstance(self.weight_matrix, np.ndarray):
            W = self.weight_matrix
        else:
            def process_xi(xi):
                weightf = lambda xi, xj, sigma: np.exp(-(xi-xj).dot(xi-xj)/(2*sigma**2))
                i = xi[0]
                xi = xi[1]
                # f = multivariate_normal(mean=xi, cov=cov)
                # w = [f.pdf([X[j]]) if i != j else 0 for j in range(len(X))]
                w = [weightf(xi, X[j], sigma) if i != j else 0 for j in range(len(X))]
                w = [ww if ww > self.w_cut else 0 for ww in w]
                return w
            W = np.array(list(map(process_xi, enumerate(X))))
            # print("w shape", W.shape)
            # print("Creating weight matrix cost", datetime.now() - tic2)
            self.weight_matrix = W

        A = []
        for i in range(Nl, N):
            a = [(-np.sum(W[i]) if i == j else W[i][j]) for j in range(Nl, N)]
            A.append(a)
        A = np.array(A)

        ## Create b
        b = np.array([-np.sum([w * self._Y0[j] if j < Nl else 0 for j, w in enumerate(W[i])]) for i in range(Nl, N)])

        return A, b

    def solve(self, method="cg", ret=False):
        A, b = self.create_linear_system()
        if method == "cg":
            U = scipy.sparse.linalg.cg(A, b)[0]
        elif method == "npsolver":
            U = np.linalg.solve(A, b)
        else:
            raise(Exception("Unknown Method"))

        U = np.array(list(self._Y0) + list(U))
        self.U = U
        Y = np.array([1 if u > self._t else 0 for u in U])
        self.Y = Y
        self._solved = True
        if ret:
            return U, Y


class MBOBinary:
    def __init__(self, X0, Y0, X1, sigma=0.5, dT=0.05, Nd=5, initial="GL",
                 weight_matrix=None, w_cut=0.001):
        # def weight_func(xi, xj, sigm):
        #     xixj = xi - xj
        #     xixj2 = np.dot(xixj, xixj)
        #     w = np.float16(np.exp(-(xixj2) / (sigm ** 2)))
        #     w = w if w > 1E-3 else 0
        #     return w

        self._X0 = X0
        self._Y0 = Y0
        self._X1 = X1
        self._solved = False
        # self.__wf = weight_func
        # self.__sigm = sigm
        self._sigma = sigma
        self._dt = dT/Nd
        self._Nd = Nd
        self.Y = list()
        self.X = list()
        self.initial = initial
        self.weight_matrix = weight_matrix
        self.w_cut = w_cut

    # @property
    # def weight_function(self):
    #     return self.__wf
    #
    # @weight_function.setter
    # def weight_function(self, f):
    #     self.__wf = f

    def solve(self):
        # weight_func = self.__wf
        # sigm = self.__sigm
        sigma = self._sigma
        dt = self._dt

        ## Initial Guest
        X = np.array(self._X0 + self._X1)
        Nl = len(self._X0)
        Nu = len(self._X1)
        N = len(X)
        if self.initial == "GL":
            print("Running GL...", end=" ")
            GL = GraphLaplacian(self._X0, self._Y0, self._X1,
                                sigma=self._sigma, weight_matrix=self.weight_matrix,
                                w_cut=self.w_cut)
            GL.solve()
            U = np.array(GL.Y[Nl:])
            self.weight_matrix = GL.weight_matrix
            print("Done..")
        elif self.initial == "0.5":
            U = np.ones(Nu) * 0.5
        n_feature = len(X[0]) if isinstance(X[0], np.ndarray) else 1

        # W = []
        # for i, x in enumerate(X):
        #     w = [weight_func(X[i], X[j], sigm) if i!=j else 0 for j in range(len(X))]
        #     W.append(w)
        # W = np.array(W)

        # try:
        #     _ = iter(cov)
        #     if cov.ndim == 1:
        #         # print("Sampe Sini")
        #         cov = np.diag(cov * cov)
        #     else:
        #         pass
        # except TypeError:
        #     cov = np.diag((cov * cov) * np.ones(n_feature))
        # W = []
        tic2 = datetime.now()

        print("Calculate weight matrix...", end=" ")
        if isinstance(self.weight_matrix, np.ndarray):
            W = self.weight_matrix
        else:
            def process_xi(xi):
                weightf = lambda xi, xj, sigma: np.exp(-(xi-xj).dot(xi-xj)/(2*sigma**2))
                i = xi[0]
                xi = xi[1]
                # f = multivariate_normal(mean=xi, cov=cov)
                # w = [f.pdf([X[j]]) if i != j else 0 for j in range(len(X))]
                w = [weightf(xi, X[j], sigma) if i != j else 0 for j in range(len(X))]
                w = [ww if ww > self.w_cut else 0 for ww in w]
                return w
            W = np.array(list(map(process_xi, enumerate(X))))
            # print("w shape", W.shape)
            # print("Creating weight matrix cost", datetime.now() - tic2)
            self.weight_matrix = W
        print("DONE")

        ## START MBO Iteration
        Unew = U
        tracking_data = list()
        tracking_data.append(
            {"k":0, "Nd":0, "State":"Initial", "Y": np.array(list(self._Y0) + list(U))}
        )

        for k in range(1000):
            for nd in range(self._Nd):
                Uold = Unew
                A = []
                for i in range(Nl, N):
                    a = [(-np.sum(W[i])*dt - 1 if i==j else W[i][j]*dt) for j in range(Nl, N)]
                    A.append(a)

                A = np.array(A)
                b = np.array([-np.sum([w * self._Y0[j] * dt if j < Nl else 0 for j, w in enumerate(W[i])])
                              - Unew[i-Nl]
                              for i in range(Nl, N)])

                Unew = scipy.sparse.linalg.cg(A, b)[0]
                error = np.dot(Unew - Uold, Unew - Uold) / np.dot(Unew, Unew)
                tracking_data.append(
                    {"k": k+1, "Nd":nd+1, "State": "Small Step", "Y": np.array(list(self._Y0) + list(Unew))}
                )
                # if error < 1e-5:
                #     break

            ## Thresholding
            Unew = np.array([1 if u >= 0.5 else 0 for u in Unew])
            tracking_data.append(
                {"k": k+1, "Nd":nd+1, "State": "After Tresholding", "Y": np.array(list(self._Y0) + list(Unew))}
            )
            Unew_err = np.array(list(self._Y0) + list(Unew))
            U_err = np.array(list(self._Y0) + list(U))
            error = np.dot(Unew_err - U_err, Unew_err - U_err) / np.dot(Unew_err, Unew_err)
            print("Interation",k,"error",error)
            if error < 1e-4:
                break
            U = Unew

        Unew = np.array(list(self._Y0) + list(Unew))
        self.Y = Unew
        self.tracking = tracking_data

