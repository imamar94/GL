import numpy as np
from .binary import GraphLaplacian
from scipy.stats import multivariate_normal
from datetime import datetime
import multiprocessing


class GLMultiClass:
    def __init__(self, X0, Y0, X1, cov=0.1):
        self.__X0 = X0
        self.__Y0 = Y0
        self.__X1 = X1
        self.__solved = False
        self.__cov = cov
        self.Y = list()
        self.U = list()
        self.weight_matrix = False
        self.feature_important = False

    # @property
    # def weight_function(self):
    #     return self.__wf
    #
    # @weight_function.setter
    # def weight_function(self, f):
    #     self.__wf = f

    def run_one_label(self, label):
        # print("Running for label", label)
        tick = datetime.now()
        Y0_ = [1 if y == label else 0 for y in self.__Y0]
        cov = self.__cov
        if self.feature_important:
            if isinstance(self.weight_matrix, np.ndarray):
                self.weight_matrix = None
            feature_amd = list()
            feature_index = list(range(len(self.__X0[0])))
            for feature_n in feature_index:
                xn = [x[feature_n] for x in self.__X0]
                label_set = np.mean(np.array(xn)[[y == label for y in self.__Y0]])
                avg_mean_distance = list()
                for l in set(self.__Y0):
                    if l == label:
                        pass
                    else:
                        distance = np.abs(np.mean(label_set) - np.mean(np.array(xn)[[y == l for y in self.__Y0]]))
                        avg_mean_distance.append(distance)
                avg_mean_distance = np.mean(avg_mean_distance)
                feature_amd.append(avg_mean_distance)
                # print("avg mean distance for feature", feature_n, ":", avg_mean_distance)
            normalized_amd = feature_amd / max(feature_amd)
            normalized_amd = np.array([x if x > 0.01 else 0.01 for x in normalized_amd])
        else:
            normalized_amd = []
        GL = GraphLaplacian(self.__X0, Y0_, self.__X1,
                                    cov=cov, weight_matrix=self.weight_matrix)
        GL.solve(method="npsolver", feature_important=normalized_amd)
        self.weight_matrix = GL.weight_matrix
        # print("Running time", datetime.now() - tick)
        return GL.U

    def solve(self, feature_important=False):
        Uk = list()
        pool = multiprocessing.Pool(4)
        labels = list(set(self.__Y0))
        if feature_important:
            self.feature_important = True
        Uk = [self.run_one_label(labels[0])]

        Uk += pool.map(self.run_one_label, labels[1:])
        # for label in set(self.__Y0):
        #     print("Running for label", label)
        #     tick = datetime.now()
        #     Y0_ = [1 if y == label else 0 for y in self.__Y0]
        #     GL = GraphLaplacianModified(self.__X0, Y0_, self.__X1,
        #                         cov=self.__cov)
        #     # GL.weight_function = self.__wf
        #     GL.solve(method="npsolver")
        #     Uk.append(GL.U)
        #     print("Running time", datetime.now() - tick)

        self.U = np.array(Uk).T
        self.Y = np.array([np.argmax(u) for u in np.array(Uk).T])


class MBOMultiClass:
    def __init__(self, X0, Y0, X1, cov=0.5, dT=0.05, Nd=3):
        # def weight_func(xi, xj, sigm):
        #     xixj = xi - xj
        #     xixj2 = np.dot(xixj, xixj)
        #     w = np.float16(np.exp(-(xixj2) / (sigm ** 2)))
        #     w = w if w > 1E-3 else 0
        #     return w

        self.__X0 = X0
        self.__Y0 = Y0
        self.__X1 = X1
        self.__solved = False
        # self.__wf = weight_func
        # self.__sigm = sigm
        self.__cov = cov
        self.__dt = dT/Nd
        self.__Nd = Nd
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
        cov = self.__cov
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
        n_feature = len(X[0])

        # W = []
        # for i, x in enumerate(X):
        #     w = [weight_func(X[i], X[j], sigm) if i != j else 0 for j in range(len(X))]
        #     W.append(w)
        # W = np.array(W)

        try:
            _ = iter(cov)
            if cov.ndim == 1:
                # print("Sampe Sini")
                cov = np.diag(cov * cov)
            else:
                pass
        except TypeError:
            cov = np.diag((cov * cov) * np.ones(n_feature))
        # W = []
        tic2 = datetime.now()

        def process_xi(xi):
            i = xi[0]
            xi = xi[1]
            f = multivariate_normal(mean=xi, cov=cov)
            w = [f.pdf([X[j]]) if i != j else 0 for j in range(len(X))]
            return w
        W = np.array(list(map(process_xi, enumerate(X))))
        # print("Creating weight matrix cost", datetime.now() - tic2)

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
