import numpy as np
from .binary import GraphLaplacian
from scipy.stats import multivariate_normal
from datetime import datetime
import multiprocessing


class GLMultiClass:
    def __init__(self, X0, Y0, X1, sigma=0.1, weight_matrix=False, w_cut=0.001):
        self._X0 = X0
        self._Y0 = Y0
        self._X1 = X1
        self._solved = False
        self._sigma = sigma
        self.Y = list()
        self.U = list()
        self.weight_matrix = weight_matrix
        self.feature_important = False
        self.w_cut = w_cut

    def run_one_label(self, label):
        # print("Running for label", label)
        tick = datetime.now()
        Y0_ = [1 if y == label else 0 for y in self._Y0]
        sigma = self._sigma
        GL = GraphLaplacian(self._X0, Y0_, self._X1,
                            sigma=sigma, weight_matrix=self.weight_matrix,
                            w_cut=self.w_cut)
        GL.solve(method="npsolver")
        self.weight_matrix = GL.weight_matrix
        # print("Running time", datetime.now() - tick)
        return GL.U

    def solve(self):
        pool = multiprocessing.Pool(4)
        labels = list(set(self._Y0))
        Uk = [self.run_one_label(labels[0])]
        Uk += pool.map(self.run_one_label, labels[1:])
        pool.close()
        pool.join()

        self.U = np.array(Uk).T
        self.Y = np.array([np.argmax(u) for u in np.array(Uk).T])


class MBOMultiClass:
    def __init__(self, X0, Y0, X1, sigma=0.5, dT=0.05, Nd=3,
                 weight_matrix=False, w_cut=0.001):
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
        self.weight_matrix = weight_matrix
        self.w_cut = w_cut
        self.Y = list()
        self.X = list()

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

        ## Initial Guess
        X = np.array(self._X0 + self._X1)
        Nl = len(self._X0)
        Nu = len(self._X1)
        N = len(X)
        n_classes = len(np.unique(self._Y0))
        temp = range(n_classes)
        Y0t = [[1 if x == y else 0 for x in temp] for y in self._Y0]
        U = np.ones([Nu, n_classes])*0.5
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
        tic2 = datetime.now()

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
        print("Creating weight matrix cost", datetime.now() - tic2)

        ## START MBO Iteration
        Unew = U
        for k in range(100):
            for nd in range(self._Nd):
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
                # error = np.sum((Unew - Uold) * (Unew - Uold)) / np.sum(Unew * Unew)
                # if error < 1e-5:
                #     break

            ## Thresholding
            # Unew_err = np.array(list(self.__Y0) + [np.argmax(u) for u in Unew])
            # U_err = np.array(list(self.__Y0) + [np.argmax(u) for u in U])
            Unew_err = list(Y0t) + list(Unew)
            U_err = list(Y0t) + list(U)
            Unew_err = np.array([np.argmax(u) for u in Unew_err])
            U_err = np.array([np.argmax(u) for u in U_err])
            error = np.dot(Unew_err - U_err, Unew_err - U_err) / np.dot(Unew_err, Unew_err)
            print(error)
            if error < 5e-3:
                break
            U = Unew

        Y = list(Y0t) + list(Unew)
        self.Y = np.array([np.argmax(u) for u in Y])
