import numpy as np
import Orange
import Orange.classification
import Orange.evaluation
import Orange.preprocess
from scipy.optimize import fmin_l_bfgs_b
import sklearn


def add_ones(X):
    """Adds a column of ones to the data matrix."""
    return np.column_stack((np.ones(len(X)), X))


def g(z):
    """Logistic function."""
    return 1/(1+np.exp(-z))

log = lambda x: np.log(x + 1e-8)


class NeuralNetLearner(Orange.classification.Learner):
    """Neural network learner"""
    def __init__(self, arch, lambda_=1e-5):
        super().__init__()

        # network architecture and shape of thetas
        self.arch = arch
        self.theta_shape = np.array([(arch[i]+1, arch[i+1])
                                     for i in range(len(arch)-1)])
        ind = np.array([arch[i+1] * (arch[i]+1) for i in range(len(arch)-1)])
        self.theta_len = sum(ind)
        self.theta_ind = np.cumsum(ind)
        self.b = np.hstack(np.array([0]*r+[1]*(c-1)*r)
                           for (c, r) in self.theta_shape)

        self.lambda_ = lambda_  # degree of regularization
        self.name = "ann"

        self.X, self.y, self.m = None, None, None

        self.preprocessors += [
            # Orange.preprocess.preprocess.Continuize(),
            Orange.preprocess.fss.RemoveNaNColumns(),
            Orange.preprocess.preprocess.SklImpute()
        ]

    def shape_thetas(self, thetas):
        thetas = np.split(thetas, self.theta_ind)
        thetas = [thetas[i].reshape(shape)
                  for i, shape in enumerate(self.theta_shape)]
        return thetas

    def init_thetas(self, epsilon=1e-1):
        """Initialization of model parameters."""
        return np.random.rand(self.theta_len) * 2 * epsilon - epsilon

    def h(self, a, thetas):
        """Feed forward, prediction."""
        thetas = self.shape_thetas(thetas)
        for theta in thetas:
            a = g(add_ones(a).dot(theta))
        return a

    def J(self, thetas):
        err = np.sum((np.ravel(self.h(self.X, thetas)) - self.y) ** 2) / 2
        reg = np.sum(thetas ** 2 * self.b)
        return err / self.m + self.lambda_ * reg / 2 / self.m

    def grad_approx(self, thetas):
        """Approximate computation of cost function gradient."""
        e = 1e-5
        return np.array([(self.J(thetas + row) - self.J(thetas - row)) / (2 * e)
                         for row in np.identity(len(thetas)) * e])

    def backprop(self, flat_thetas):
        thetas = self.shape_thetas(flat_thetas)
        # feed forward for activations
        a = [self.X]
        for theta in thetas:
            a.append(g(add_ones(a[-1]).dot(theta)))

        # backprop for error and gradient
        y = self.y.reshape((self.y.shape[0], 1))
        d = -(y - a[-1]) * (a[-1] * (1-a[-1]))
        D = np.ravel(add_ones(a[-2]).T.dot(d))
        for i in range(2, len(self.arch)):
            d = d.dot(thetas[-i+1].T)[:, 1:] * (a[-i] * (1-a[-i]))
            D = np.hstack((np.ravel(add_ones(a[-i-1]).T.dot(d)), D))

        return (1/self.m) * D + self.lambda_ * flat_thetas * self.b / self.m
        # return (1/self.m) * D + self.lambda_ * flat_thetas / self.m

    def callback(self, thetas):
        print("J", self.J(thetas))

    def fit(self, X, y, W=None):
        self.X, self.y = X, y
        self.m = self.X.shape[0]
        thetas = self.init_thetas()

        thetas, fmin, info = fmin_l_bfgs_b(self.J, thetas, self.backprop,
                                           # callback=self.callback,
                                           )

        model = NeuralNetClassifier(self.domain, thetas)
        model.h = self.h
        return model

    def test(self, data):
        self.X, self.y = data.X, data.Y
        thetas = self.init_thetas()
        # thetas = np.array([-25,  15,  20, -20,  20, -20, -10,  20,  20])

        y = np.array(range(len(data.domain.class_var.values)))
        y = y.astype(np.float32)
        self.y = np.tile(y, (len(data), 1))

        print(self.h(self.X[:2], thetas))

        print("J", self.J(thetas))
        # g0 = self.grad_approx(self.thetas)
        # g1 = self.backprop(self.thetas)
        # for z0, z1 in zip(g0, g1):
        #     print("%11.8f %11.8f -- %11.8f" % (z0, z1, abs(z0-z1)))


class NeuralNetClassifier(Orange.classification.Model):
    """Multi-class classifier based on a set of binary classifiers."""
    def __init__(self, domain, thetas):
        super().__init__(domain)
        self.thetas = thetas  # model parameters

    def predict(self, X):
        y_hat = np.ravel(self.h(X, self.thetas))
        return np.vstack((1-y_hat, y_hat)).T


# data = Orange.data.Table("xor.tab")
data = Orange.data.Table("iris.tab")
# data = Orange.data.Table("voting")
# data = Orange.data.Table("adult_sample")
# data = Orange.data.Table("titanic")
# data = Orange.data.Table("promoters")
# data = Orange.data.Table(np.array([[-1, 1]]), np.array([[0]]))

data = Orange.preprocess.Impute(data)
data = Orange.preprocess.Continuize(data)
data.X = data.X.astype(np.float32)
data.Y = data.Y.astype(np.float32)

scaler = sklearn.preprocessing.StandardScaler()
data.X = scaler.fit_transform(data.X)

n = data.X.shape[1]
ann = NeuralNetLearner((n, 2, 3), lambda_=0.01)

# model = ann(data)
ann.test(data)

# rf = Orange.classification.SimpleRandomForestLearner(n_estimators=10)
# lr = Orange.classification.LogisticRegressionLearner()
#
# res = Orange.evaluation.CrossValidation(data, [ann], k=3)
# scores = Orange.evaluation.AUC(res)
# print(scores)