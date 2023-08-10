#!/usr/bin/env python
# coding: utf-8
# from sklearn.linear_model import Ridge
import os
import numpy as np
import numpy as np
import numpy as np
import torch
# from scipy import linalg
from sklearn.mixture import GaussianMixture 
from sklearn.decomposition import PCA
# from utils import iterative_A
from multiprocessing import Pool
# from pathos.multiprocessing import ProcessingPool as Pool
from joblib import Parallel, delayed

def _cov(X, shrinkage=-1):
    emp_cov = np.cov(np.asarray(X).T, bias=1)
    if shrinkage < 0:
        return emp_cov
    n_features = emp_cov.shape[0]
    mu = np.trace(emp_cov) / n_features
    shrunk_cov = (1.0 - shrinkage) * emp_cov
    shrunk_cov.flat[:: n_features + 1] += shrinkage * mu
    return shrunk_cov


def softmax(X, copy=True):
    if copy:
        X = np.copy(X)
    max_prob = np.max(X, axis=1).reshape((-1, 1))
    X -= max_prob
    np.exp(X, X)
    sum_prob = np.sum(X, axis=1).reshape((-1, 1))
    X /= sum_prob
    return X


def _class_means(X, y):
    """Compute class means.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.
    Returns
    -------
    means : array-like of shape (n_classes, n_features)
        Class means.
    means ： array-like of shape (n_classes, n_features)
        Outer classes means.
    """
    classes, y = np.unique(y, return_inverse=True)
    cnt = np.bincount(y)
    means = np.zeros(shape=(len(classes), X.shape[1]))
    np.add.at(means, y, X)
    means /= cnt[:, None]

    means_ = np.zeros(shape=(len(classes), X.shape[1]))
    for i in range(len(classes)):
        means_[i] = (np.sum(means, axis=0) - means[i]) / (len(classes) - 1)    
    return means, means_


def split_data(data: np.ndarray, percent_train: float):
    split = data.shape[0] - int(percent_train * data.shape[0])
    return data[:split], data[split:]


def feature_reduce(features: np.ndarray, f: int=None):
    """
        Use PCA to reduce the dimensionality of the features.
        If f is none, return the original features.
        If f < features.shape[0], default f to be the shape.
	"""
    if f is None:
        return features
    if f > features.shape[0]:
        f = features.shape[0]
    
    return sklearn.decomposition.PCA(
        n_components=f,
        svd_solver='randomized',
        random_state=1919,
        iterated_power=1).fit_transform(features)


class TransferabilityMethod:	
    def __call__(self, 
        features: np.ndarray, y: np.ndarray,
                ) -> float:
        self.features = features		
        self.y = y
        return self.forward()

    def forward(self) -> float:
        raise NotImplementedError


class PARC(TransferabilityMethod):
	
    def __init__(self, n_dims: int=None, fmt: str=''):
        self.n_dims = n_dims
        self.fmt = fmt

    def forward(self):
        self.features = feature_reduce(self.features, self.n_dims)
        
        num_classes = len(np.unique(self.y, return_inverse=True)[0])
        labels = np.eye(num_classes)[self.y] if self.y.ndim == 1 else self.y

        return self.get_parc_correlation(self.features, labels)

    def get_parc_correlation(self, feats1, labels2):
        scaler = sklearn.preprocessing.StandardScaler()

        feats1  = scaler.fit_transform(feats1)

        rdm1 = 1 - np.corrcoef(feats1)
        rdm2 = 1 - np.corrcoef(labels2)
        
        lt_rdm1 = self.get_lowertri(rdm1)
        lt_rdm2 = self.get_lowertri(rdm2)
        
        return scipy.stats.spearmanr(lt_rdm1, lt_rdm2)[0] * 100

    def get_lowertri(self, rdm):
        num_conditions = rdm.shape[0]
        return rdm[np.triu_indices(num_conditions, 1)]


class SFDA():
    def __init__(self, shrinkage=None, priors=None, n_components=None):
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        
    def _solve_eigen(self, X, y, shrinkage):
        classes, y = np.unique(y, return_inverse=True)
        cnt = np.bincount(y)
        means = np.zeros(shape=(len(classes), X.shape[1]))
        np.add.at(means, y, X)
        means /= cnt[:, None]
        self.means_ = means
                
        cov = np.zeros(shape=(X.shape[1], X.shape[1]))
        for idx, group in enumerate(classes):
            Xg = X[y == group, :]
            cov += self.priors_[idx] * np.atleast_2d(_cov(Xg))
        self.covariance_ = cov

        Sw = self.covariance_  # within scatter
        if self.shrinkage is None:
            # adaptive regularization strength
            largest_evals_w = iterative_A(Sw, max_iterations=3)
            shrinkage = max(np.exp(-5 * largest_evals_w), 1e-10)
            self.shrinkage = shrinkage
        else:
            # given regularization strength
            shrinkage = self.shrinkage
        print("Shrinkage: {}".format(shrinkage))
        # between scatter
        St = _cov(X, shrinkage=self.shrinkage) 

        # add regularization on within scatter   
        n_features = Sw.shape[0]
        mu = np.trace(Sw) / n_features
        shrunk_Sw = (1.0 - self.shrinkage) * Sw
        shrunk_Sw.flat[:: n_features + 1] += self.shrinkage * mu

        Sb = St - shrunk_Sw  # between scatter

        evals, evecs = linalg.eigh(Sb, shrunk_Sw)
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors

        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )

    def fit(self, X, y):
        '''
        X: input features, N x D
        y: labels, N

        '''
        self.classes_ = np.unique(y)
        #n_samples, _ = X.shape
        n_classes = len(self.classes_)

        max_components = min(len(self.classes_) - 1, X.shape[1])

        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than min(n_features, n_classes - 1)."
                )
            self._max_components = self.n_components

        _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
        self.priors_ = np.bincount(y_t) / float(len(y))
        self._solve_eigen(X, y, shrinkage=self.shrinkage,)

        return self
    
    def transform(self, X):
        # project X onto Fisher Space
        X_new = np.dot(X, self.scalings_)
        return X_new[:, : self._max_components]

    def predict_proba(self, X):
        scores = np.dot(X, self.coef_.T) + self.intercept_   ##计算分数
        return softmax(scores)


def each_evidence(y_, f, fh, v, s, vh, N, D):
    """
    compute the maximum evidence for each class
    """
    epsilon = 1e-5
    alpha = 1.0
    beta = 1.0
    lam = alpha / beta
    tmp = (vh @ (f @ np.ascontiguousarray(y_)))
    for _ in range(11):
        # should converge after at most 10 steps
        # typically converge after two or three steps
        gamma = (s / (s + lam)).sum()
        # A = v @ np.diag(alpha + beta * s) @ v.transpose() # no need to compute A
        # A_inv = v @ np.diag(1.0 / (alpha + beta * s)) @ v.transpose() # no need to compute A_inv
        m = v @ (tmp * beta / (alpha + beta * s))
        alpha_de = (m * m).sum()
        alpha = gamma / (alpha_de + epsilon)
        beta_de = ((y_ - fh @ m) ** 2).sum()
        beta = (N - gamma) / (beta_de + epsilon)
        new_lam = alpha / beta
        if np.abs(new_lam - lam) / lam < 0.01:
            break
        lam = new_lam
    evidence = D / 2.0 * np.log(alpha) \
               + N / 2.0 * np.log(beta) \
               - 0.5 * np.sum(np.log(alpha + beta * s)) \
               - beta / 2.0 * (beta_de + epsilon) \
               - alpha / 2.0 * (alpha_de + epsilon) \
               - N / 2.0 * np.log(2 * np.pi)
    return evidence / N, alpha, beta, m


def truncated_svd(x):
    u, s, vh = np.linalg.svd(x.transpose() @ x)
    s = np.sqrt(s)
    u_times_sigma = x @ vh.transpose()
    k = np.sum((s > 1e-10) * 1)  # rank of f
    s = s.reshape(-1, 1)
    s = s[:k]
    vh = vh[:k]
    u = u_times_sigma[:, :k] / s.reshape(1, -1)
    return u, s, vh


class LogME(object):
    def __init__(self, regression=False):
        """
            :param regression: whether regression
        """
        self.regression = regression
        self.fitted = False
        self.reset()

    def reset(self):
        self.num_dim = 0
        self.alphas = []  # alpha for each class / dimension
        self.betas = []  # beta for each class / dimension
        # self.ms.shape --> [C, D]
        self.ms = []  # m for each class / dimension

    def _fit_icml(self, f: np.ndarray, y: np.ndarray):
        """
        LogME calculation proposed in the ICML 2021 paper
        "LogME: Practical Assessment of Pre-trained Models for Transfer Learning"
        at http://proceedings.mlr.press/v139/you21b.html
        """
        fh = f
        f = f.transpose()
        D, N = f.shape
        v, s, vh = np.linalg.svd(f @ fh, full_matrices=True)

        evidences = []
        self.num_dim = y.shape[1] if self.regression else int(y.max() + 1)
        for i in range(self.num_dim):
            y_ = y[:, i] if self.regression else (y == i).astype(np.float64)
            evidence, alpha, beta, m = each_evidence(y_, f, fh, v, s, vh, N, D)
            evidences.append(evidence)
            self.alphas.append(alpha)
            self.betas.append(beta)
            self.ms.append(m)
        self.ms = np.stack(self.ms)
        return np.mean(evidences)

    def _fit_fixed_point(self, f: np.ndarray, y: np.ndarray):
        """
        LogME calculation proposed in the arxiv 2021 paper
        "Ranking and Tuning Pre-trained Models: A New Paradigm of Exploiting Model Hubs"
        at https://arxiv.org/abs/2110.10545
        """
        # k = min(N, D)
        N, D = f.shape  

        # direct SVD may be expensive
        if N > D: 
            u, s, vh = truncated_svd(f)
        else:
            u, s, vh = np.linalg.svd(f, full_matrices=False)
        # u.shape = N x k, s.shape = k, vh.shape = k x D
        s = s.reshape(-1, 1)
        sigma = (s ** 2)

        evidences = []
        self.num_dim = y.shape[1] if self.regression else int(y.max() + 1)
        for i in range(self.num_dim):
            y_ = y[:, i] if self.regression else (y == i).astype(np.float64)
            y_ = y_.reshape(-1, 1)
            
            # x has shape [k, 1], but actually x should have shape [N, 1]
            
            x = u.T @ y_  

            x2 = x ** 2
            # if k < N, we compute sum of xi for 0 singular values directly
            res_x2 = (y_ ** 2).sum() - x2.sum()  

            alpha, beta = 1.0, 1.0
            for _ in range(11):
                t = alpha / beta
                gamma = (sigma / (sigma + t)).sum()
                m2 = (sigma * x2 / ((t + sigma) ** 2)).sum()
                res2 = (x2 / ((1 + sigma / t) ** 2)).sum() + res_x2
                alpha = gamma / (m2 + 1e-5)
                beta = (N - gamma) / (res2 + 1e-5)
                t_ = alpha / beta
                evidence = D / 2.0 * np.log(alpha) \
                           + N / 2.0 * np.log(beta) \
                           - 0.5 * np.sum(np.log(alpha + beta * sigma)) \
                           - beta / 2.0 * res2 \
                           - alpha / 2.0 * m2 \
                           - N / 2.0 * np.log(2 * np.pi)
                evidence /= N
                if abs(t_ - t) / t <= 1e-3:  # abs(t_ - t) <= 1e-5 or abs(1 / t_ - 1 / t) <= 1e-5:
                    break
            evidence = D / 2.0 * np.log(alpha) \
                       + N / 2.0 * np.log(beta) \
                       - 0.5 * np.sum(np.log(alpha + beta * sigma)) \
                       - beta / 2.0 * res2 \
                       - alpha / 2.0 * m2 \
                       - N / 2.0 * np.log(2 * np.pi)
            evidence /= N
            m = 1.0 / (t + sigma) * s * x
            m = (vh.T @ m).reshape(-1)
            evidences.append(evidence)
            self.alphas.append(alpha)
            self.betas.append(beta)
            self.ms.append(m)
        self.ms = np.stack(self.ms)
        return np.mean(evidences)

    _fit = _fit_fixed_point
    #_fit = _fit_icml

    def fit(self, f: np.ndarray, y: np.ndarray):
        """
        :param f: [N, F], feature matrix from pre-trained model
        :param y: target labels.
            For classification, y has shape [N] with element in [0, C_t).
            For regression, y has shape [N, C] with C regression-labels

        :return: LogME score (how well f can fit y directly)
        """
        if self.fitted:
            warnings.warn('re-fitting for new data. old parameters cleared.')
            self.reset()
        else:
            self.fitted = True
        f = f.astype(np.float64)
        if self.regression:
            y = y.astype(np.float64)
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
        return self._fit(f, y)

    def predict(self, f: np.ndarray):
        """
        :param f: [N, F], feature matrix
        :return: prediction, return shape [N, X]
        """
        if not self.fitted:
            raise RuntimeError("not fitted, please call fit first")
        f = f.astype(np.float64)
        logits = f @ self.ms.T
        if self.regression:
            return logits
        prob = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)  
        # return np.argmax(logits, axis=-1)
        return prob


def LEEP(X, y, model_name='resnet50'):

    n = len(y)
    num_classes = len(np.unique(y))

    # read classifier
    # Group1: model_name, fc_name, model_ckpt
    ckpt_models = {
        'densenet121': ['classifier.weight', '/CNN_models/classifier/checkpoints/densenet121-a639ec97.pth'],
        'densenet169': ['classifier.weight', '/CNN_models/classifier/checkpoints/densenet169-b2777c0a.pth'],
        'densenet201': ['classifier.weight', '/CNN_models/classifier/checkpoints/densenet201-c1103571.pth'],
        'resnet34': ['fc.weight', '/CNN_models/classifier/checkpoints/resnet34-333f7ec4.pth'],
        'resnet50': ['fc.weight', '/CNN_models/classifier/checkpoints/resnet50-19c8e357.pth'],
        'resnet101': ['fc.weight', '/CNN_models/classifier/checkpoints/resnet101-5d3b4d8f.pth'],
        'resnet152': ['fc.weight', '/CNN_models/classifier/checkpoints/resnet152-b121ed2d.pth'],
        'mnasnet1_0': ['classifier.1.weight', '/CNN_models/classifier/checkpoints/mnasnet1.0_top1_73.512-f206786ef8.pth'],
        'mobilenet_v2': ['classifier.1.weight', '/CNN_models/classifier/checkpoints/mobilenet_v2-b0353104.pth'],
        'googlenet': ['fc.weight', '/CNN_models/classifier/checkpoints/googlenet-1378be20.pth'],
        'inception_v3': ['fc.weight', '/CNN_models/classifier/checkpoints/inception_v3_google-1a9a5a14.pth'],
    }

    # which need to be trained is you use LEEP.
	
    ckpt_loc = ckpt_models[model_name][1]
    fc_weight = ckpt_models[model_name][0]
    fc_bias = fc_weight.replace('weight', 'bias')
    ckpt = torch.load(ckpt_loc, map_location='cpu')
    fc_weight = ckpt[fc_weight].detach().numpy()
    fc_bias = ckpt[fc_bias].detach().numpy()

    # p(z|x), z is source label
    prob = np.dot(X, fc_weight.T) + fc_bias
    prob = softmax(prob)   # p(z|x), N x C(source)

    pyz = np.zeros((num_classes, 1000))  # C(source) = 1000
    for y_ in range(num_classes):
        indices = np.where(y == y_)[0]
        filter_ = np.take(prob, indices, axis=0) 
        pyz[y_] = np.sum(filter_, axis=0) / n
    
    pz = np.sum(pyz, axis=0)     # marginal probability
    py_z = pyz / pz              # conditional probability, C x C(source)
    py_x = np.dot(prob, py_z.T)  # N x C

    # leep = E[p(y|x)]
    leep_score = np.sum(py_x[np.arange(n), y]) / n
    return leep_score

import sklearn.decomposition as sd
def NLEEP(X, y, component_ratio=5):

    n = len(y)
    num_classes = len(np.unique(y))
    # PCA: keep 80% energy
    pca_80 = PCA(n_components=0.8)
    pca_80.fit(X)
    X_pca_80 = pca_80.transform(X)

    # GMM: n_components = component_ratio * class number
    n_components_num = component_ratio * num_classes
    gmm = GaussianMixture(n_components= n_components_num).fit(X_pca_80)
    prob = gmm.predict_proba(X_pca_80)  # p(z|x)
    
    # NLEEP
    pyz = np.zeros((num_classes, n_components_num))
    for y_ in range(num_classes):
        indices = np.where(y == y_)[0]
        filter_ = np.take(prob, indices, axis=0) 
        pyz[y_] = np.sum(filter_, axis=0) / n   
    pz = np.sum(pyz, axis=0)    
    py_z = pyz / pz             
    py_x = np.dot(prob, py_z.T) 

    # nleep_score
    nleep_score = np.sum(py_x[np.arange(n), y]) / n
    return nleep_score


def LogME_Score(X, y):

    logme = LogME(regression=True)
    score = logme.fit(X, y)
    return score


def SFDA_Score(X, y):

    n = len(y)
    num_classes = len(np.unique(y))
    # X_features, y_labels = np.load(model_npy_feature), np.load(model_npy_label)
    # SFDA_Score(X_features, y_labels) 
    SFDA_first = SFDA()
    prob = SFDA_first.fit(X, y).predict_proba(X)  # p(y|x)
    
    # soften the probability using softmax for meaningful confidential mixture
    prob = np.exp(prob) / np.exp(prob).sum(axis=1, keepdims=True) 
    means, means_ = _class_means(X, y)  # class means, outer classes means
    

    
    # ConfMix
    for y_ in range(num_classes):
        indices = np.where(y == y_)[0]
        y_prob = np.take(prob, indices, axis=0)
        y_prob = y_prob[:, y_]  # probability of correctly classifying x with label y        
        X[indices] = y_prob.reshape(len(y_prob), 1) * X[indices] + \
                            (1 - y_prob.reshape(len(y_prob), 1)) * means_[y_]
    
    SFDA_second = SFDA(shrinkage=SFDA_first.shrinkage)
    prob = SFDA_second.fit(X, y).predict_proba(X)   # n * num_cls

    # leep = E[p(y|x)]. Note: the log function is ignored in case of instability.
    sfda_score = np.sum(prob[np.arange(n), y]) / n
    return sfda_score


def PARC_Score(X, y, ratio=2):
    
    num_sample, feature_dim = X.shape
    ndims = 32 if ratio > 1 else int(feature_dim * ratio)  # feature reduction dimension

    if num_sample > 15000:
        from utils_cr import initLabeled
        p = 15000.0 / num_sample
        labeled_index = initLabeled(y, p=p)
        features = X[labeled_index]
        targets = X[labeled_index]
        print("data are sampled to {}".format(features.shape))

    method = PARC(n_dims = ndims)
    parc_score = method(features=X, y=y)

    return parc_score


def discretize_vector(vec, num_buckets=47):
    # 计算每一块中应该包含的元素数量
    num_bins = num_buckets
    bin_size = len(vec) // num_bins
    # print(bin_size)
    # 对原始向量进行排序
    sorted_vec = vec
    # print(sorted_vec)
    # print(sorted_vec[122],sorted_vec[123])
    # 初始化结果列表
    # print(sorted_vec == vec)
    result = [0] * len(vec)
    
    # 遍历每一块
    for i in range(num_bins):
        # 计算当前块的起始和结束位置
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size
        
        
        # 如果不是最后一块，且剩余元素数量不足以填满一整块，则将多余的元素加入到前面的块中
        if i < num_bins - 1:
            if len(vec) - end_idx < bin_size:
                end_idx = len(vec)
        else:
        # if i == num_bins - 1:
            end_index = len(vec)
        # discretized_vector[start_index:end_index] = i

        
        # 将当前块中的元素映射为对应的索引值
        for j in range(start_idx, end_idx):
            # print(i,j)
            result[j] = i
    
    return np.array(result)

def discretize_vector2(vector, num_buckets):
    min_val = np.min(vector)
    max_val = np.max(vector)
    bucket_width = (max_val - min_val) / num_buckets
    
    bucket_indices = ((vector - min_val) / bucket_width).astype(int)
    
    return bucket_indices

def coding_rate(Z, eps=1e-4): 
    n, d = Z.shape
    # print(n,d)
    # print(Z.min())
    (_, rate) = np.linalg.slogdet((np.eye(d) + 1 / (n*eps)*Z.transpose()@Z))
    return 0.5*rate

def sort_with_index(array):
    """
    返回一个按照 array 排序后的索引数组
    """
    return np.argsort(array)

def Transrate(Z, y, eps=1e-4): 
    Z = Z - np.mean(Z, axis=0, keepdims=True)
    RZ = coding_rate(Z, eps) 
    RZY = 0.
    K= int(y.max() + 1) 
    # print(K)
    # score = 0
    for i in range(K):
        # print(i,'i')
        tmp_Z = Z[(y == i).flatten()]
        # print(tmp_Z,i)
        RZY += coding_rate(tmp_Z, eps) 
    return (RZ - RZY / K)


def Transrate_multi(Z, Y, eps=1e-4): 

    RZ = coding_rate(Z, eps) 
    RZY = 0.
    N,dim = Y.shape
    print(N,dim,'y.shape')
    Y = Y.T

    def process_dim(Z,y):
        # print(y.max())
        y_perdim = y
        num_bins = 50
        y_perdim_regression = discretize_vector(y_perdim, num_bins)
        K= int(y_perdim_regression.max() + 1) 
        RZY = 0
        for i in range(K):
            tmp_Z = Z[(y_perdim_regression == i).flatten()]
            RZY += coding_rate(tmp_Z, eps) 
        # print((RZ - RZY / K))
        return (RZ - RZY / K)
    # n = 10   #regression2classification
    # score = 0
    results = Parallel(n_jobs=-1)(delayed(process_dim)(Z, y) for y in Y)
    total = np.sum(results)
    return total / dim

def f(x, y):
    i = int(floor(x))
    j = int(floor(y))
    x_frac = x - i
    y_frac = y - j
    h = hilbert(x_frac, y_frac)
    return h + (i + j) * (1 + sqrt(2))

def convert2T(X,Y):
    d1,N = X.shape
    d2,N = Y.shape
    T = np.zeros([d1*d2,N*N])
    for t1 in range(d1):
        for t2 in range(d2):
            for i in range(N):
                for j in range(N):
                    index_i = t1*d2 + t2
                    index_j = i*N + j
                    T[index_i][index_j] = X[t1][i] + Y[t2][j]*Y[t2][j]
    return T

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def EMMS_optimal(Z,Y):
    N,d1 = Z.shape
    N,d2 = Y.shape
    # print(Z.shape,Y.shape)
    beta = []
    score = 0
    import numpy as np

    # 构造特征矩阵 X
    # X = np.random.rand(N, D1)

    # 对 X 在第 D1 维度上进行标准化，加入平滑项 1e-8 避免除0错误
    Z_mean = np.mean(Z, axis=0)
    Z_std = np.std(Z, axis=0)
    epsilon = 1e-8
    Z = (Z - Z_mean) / (Z_std + epsilon)

    Y_mean = np.mean(Y, axis=0)
    Y_std = np.std(Y, axis=0)
    epsilon = 1e-8
    Y = (Y - Y_mean) / (Y_std + epsilon)


    coefficients, residuals, rank, singular_values = np.linalg.lstsq(Z, Y, rcond = None)
    # residuals = np.sqrt(np.sum(residuals, axis=1))
    # print(residuals,residuals.shape,coefficients.shape)
    score = sum(residuals) / d2
    print(score)

    return 1 / (score + 0.000001)

import numpy as np

def softmax_t(x, temperature=1.0):
    """带有temperature参数的softmax函数"""
    x = np.asarray(x) / temperature
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def softmax1(x, temperature=1.0):
    """Compute softmax values for each row of x."""
    exp_x = np.exp(x / temperature)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# import numpy as np
from numpy.linalg import lstsq


def sparsemax(z):
    """forward pass for sparsemax
    this will process a 2d-array $z$, where axis 1 (each row) is assumed to be
    the the z-vector.
    """

    # sort z
    z_sorted = np.sort(z, axis=1)[:, ::-1]

    # calculate k(z)
    z_cumsum = np.cumsum(z_sorted, axis=1)
    k = np.arange(1, z.shape[1] + 1)
    z_check = 1 + k * z_sorted > z_cumsum
    # use argmax to get the index by row as .nonzero() doesn't
    # take an axis argument. np.argmax return the first index, but the last
    # index is required here, use np.flip to get the last index and
    # `z.shape[axis]` to compensate for np.flip afterwards.
    k_z = z.shape[1] - np.argmax(z_check[:, ::-1], axis=1)

    # calculate tau(z)
    tau_sum = z_cumsum[np.arange(0, z.shape[0]), k_z - 1]
    tau_z = ((tau_sum - 1) / k_z).reshape(-1, 1)

    return np.maximum(0, z - tau_z)


def EMMS(Z,Y):

    x = Z
    y = Y
    N, D2, K = y.shape
    for i in range(K):
        y_mean = np.mean(y[:,:,i] , axis=0)
        y_std = np.std(y[:,:,i], axis=0)
        epsilon = 1e-8
        y[:,:,i] = (y[:,:,i] - y_mean) / (y_std + epsilon)
    N,D1 = x.shape
    x_mean = np.mean(x , axis=0)
    x_std = np.std(x, axis=0)
    epsilon = 1e-8
    x = (x - x_mean) / (x_std + epsilon)
    lam = np.array([1/K] * K)
    w1 = 0
    lam1 = 0
    T = 0
    b = np.dot(y, lam)
    T = b
    for k in range(1):
        a = x
        w = lstsq(a, b, rcond=None)[0]
        w1 = w
        a = y.reshape(N*D2, K)
        b = np.dot(x, w).reshape(N*D2)
        lam = lstsq(a, b, rcond=None)[0]
        lam = lam.reshape(1,K)
        lam = sparsemax(lam)
        lam = lam.reshape(K,1)
        lam1 = lam
        b = np.dot(y, lam)
        b = b.reshape(N,D2)
        T = b
    y_pred = np.dot(x,w1)
    res = np.sum((y_pred - T)**2) / N*D2

    return -res

import sklearn.decomposition as sd
import sklearn.mixture as sm

def gmm_estimator(features_np_all, label_np_all):
  """Estimate the GMM posterior assignment."""
  pca_model = sd.PCA(n_components=0.8)
  pca_model.fit(features_np_all)
  features_lowdim_train = pca_model.transform(features_np_all)

  num_examples = label_np_all.shape[0]
  y_classes = max([min([label_np_all.max() + 1, int(num_examples * 0.2)]),
                   int(num_examples * 0.1)])
  clf = sm.GaussianMixture(n_components=y_classes)
  clf.fit(features_lowdim_train)
  prob_np_all_gmm = clf.predict_proba(features_lowdim_train)
  return prob_np_all_gmm, features_lowdim_train




def one_hot(a):
  b = np.zeros((a.size, a.max()+1))
  b[np.arange(a.size), a] = 1.
  return b


def calculate_pac_dir(features_np_all, label_np_all, alpha=1.):
  """Compute the PACTran-Dirichlet estimator."""
  prob_np_all,_ = gmm_estimator(features_np_all, label_np_all)
#   starttime = time.time()
  label_np_all = one_hot(label_np_all)  # [n, v]
  soft_targets_sum = np.sum(label_np_all, axis=0)  # [v]
  soft_targets_sum = np.expand_dims(soft_targets_sum, axis=1)  # [v, 1]
  a0 = alpha * soft_targets_sum / np.sum(soft_targets_sum) + 1e-10

  # initialize
  qz = prob_np_all  # [n, d]
  log_s = np.log(prob_np_all + 1e-10)  # [n, d]

  for _ in range(10):
    aw = a0 + np.sum(np.einsum("BY,BZ->BYZ", label_np_all, qz), axis=0)
    logits_qz = (log_s +
                 np.matmul(label_np_all, scipy.special.digamma(aw)) -
                 np.reshape(scipy.special.digamma(np.sum(aw, axis=0)), [1, -1]))
    log_qz = logits_qz - scipy.special.logsumexp(
        logits_qz, axis=-1, keepdims=True)
    qz = np.exp(log_qz)

  log_c0 = scipy.special.loggamma(np.sum(a0)) - np.sum(
      scipy.special.loggamma(a0))
  log_c = scipy.special.loggamma(np.sum(aw, axis=0)) - np.sum(
      scipy.special.loggamma(aw), axis=0)

  pac_dir = np.sum(
      log_c0 - log_c - np.sum(qz * (log_qz - log_s), axis=0))
  pac_dir = -pac_dir / label_np_all.size
  return pac_dir


def calculate_pac_gamma(features_np_all, label_np_all, alpha=1.):
    """Compute the PAC-Gamma estimator."""
    prob_np_all,_ = gmm_estimator(features_np_all, label_np_all)
    #   starttime = time.time()
    label_np_all = one_hot(label_np_all)  # [n, v]
    soft_targets_sum = np.sum(label_np_all, axis=0)  # [v]
    soft_targets_sum = np.expand_dims(soft_targets_sum, axis=1)  # [v, 1]

    a0 = alpha * soft_targets_sum / np.sum(soft_targets_sum) + 1e-10
    beta = 1.

    # initialize
    qz = prob_np_all  # [n, d]
    s = prob_np_all  # [n, d]
    log_s = np.log(prob_np_all + 1e-10)  # [n, d]
    aw = a0
    bw = beta
    lw = np.sum(s, axis=-1, keepdims=True) * np.sum(aw / bw)  # [n, 1]

    for _ in range(10):
        aw = a0 + np.sum(np.einsum("BY,BZ->BYZ", label_np_all, qz),
                        axis=0)  # [v, d]
        lw = np.matmul(
            s, np.expand_dims(np.sum(aw / bw, axis=0), axis=1))  # [n, 1]
        logits_qz = (
            log_s + np.matmul(label_np_all, scipy.special.digamma(aw) - np.log(bw)))
        log_qz = logits_qz - scipy.special.logsumexp(
            logits_qz, axis=-1, keepdims=True)
        qz = np.exp(log_qz)  # [n, a, d]

    pac_gamma = (
        np.sum(scipy.special.loggamma(a0) - scipy.special.loggamma(aw) +
                aw * np.log(bw) - a0 * np.log(beta)) +
        np.sum(np.sum(qz * (log_qz - log_s), axis=-1) +
                np.log(np.squeeze(lw, axis=-1)) - 1.))
    pac_gamma /= label_np_all.size
    pac_gamma += 1.
#   endtime = time.time()
    return pac_gamma


def calculate_pac_gauss(features_np_all, label_np_all,
                        lda_factor = 1.):
    """Compute the PAC_Gauss score with diagonal variance."""
    starttime = time.time()
    nclasses = label_np_all.max()+1
    label_np_all = one_hot(label_np_all)  # [n, v]
    
    mean_feature = np.mean(features_np_all, axis=0, keepdims=True)
    features_np_all -= mean_feature  # [n,k]

    bs = features_np_all.shape[0]
    kd = features_np_all.shape[-1] * nclasses
    ldas2 = lda_factor * bs  # * features_np_all.shape[-1]
    dinv = 1. / float(features_np_all.shape[-1])

    # optimizing log lik + log prior
    def pac_loss_fn(theta):
        theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses])

        w = theta[:features_np_all.shape[-1], :]
        b = theta[features_np_all.shape[-1]:, :]
        logits = np.matmul(features_np_all, w) + b

        log_qz = logits - scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        xent = np.sum(np.sum(
            label_np_all * (np.log(label_np_all + 1e-10) - log_qz), axis=-1)) / bs
        loss = xent + 0.5 * np.sum(np.square(w)) / ldas2
        return loss

    # gradient of xent + l2
    def pac_grad_fn(theta):
        theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses])

        w = theta[:features_np_all.shape[-1], :]
        b = theta[features_np_all.shape[-1]:, :]
        logits = np.matmul(features_np_all, w) + b

        grad_f = scipy.special.softmax(logits, axis=-1)  # [n, k]
        grad_f -= label_np_all
        grad_f /= bs
        grad_w = np.matmul(features_np_all.transpose(), grad_f)  # [d, k]
        grad_w += w / ldas2

        grad_b = np.sum(grad_f, axis=0, keepdims=True)  # [1, k]
        grad = np.ravel(np.concatenate([grad_w, grad_b], axis=0))
        return grad

    # 2nd gradient of theta (elementwise)
    def pac_grad2(theta):
        theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses])

        w = theta[:features_np_all.shape[-1], :]
        b = theta[features_np_all.shape[-1]:, :]
        logits = np.matmul(features_np_all, w) + b

        prob_logits = scipy.special.softmax(logits, axis=-1)  # [n, k]
        grad2_f = prob_logits - np.square(prob_logits)  # [n, k]
        xx = np.square(features_np_all)  # [n, d]

        grad2_w = np.matmul(xx.transpose(), grad2_f)  # [d, k]
        grad2_w += 1. / ldas2
        grad2_b = np.sum(grad2_f, axis=0, keepdims=True)  # [1, k]
        grad2 = np.ravel(np.concatenate([grad2_w, grad2_b], axis=0))
        return grad2

    kernel_shape = [features_np_all.shape[-1], nclasses]
    theta = np.random.normal(size=kernel_shape) * 0.03
    theta_1d = np.ravel(np.concatenate(
        [theta, np.zeros([1, nclasses])], axis=0))

    theta_1d = scipy.optimize.minimize(
        pac_loss_fn, theta_1d, method="L-BFGS-B",
        jac=pac_grad_fn,
        options=dict(maxiter=100), tol=1e-6).x

    pac_opt = pac_loss_fn(theta_1d)
    endtime_opt = time.time()

    h = pac_grad2(theta_1d)
    sigma2_inv = np.sum(h) * ldas2  / kd + 1e-10
    endtime = time.time()

    if lda_factor == 10.:
        s2s = [1000., 100.]
    elif lda_factor == 1.:
        s2s = [100., 10.]
    elif lda_factor == 0.1:
        s2s = [10., 1.]
        
    returnv = []
    for s2_factor in s2s:
        s2 = s2_factor * dinv
        pac_gauss = pac_opt + 0.5 * kd / ldas2 * s2 * np.log(
            sigma2_inv)
        
        # the first item is the pac_gauss metric
        # the second item is the linear metric (without trH)
        returnv += [("pac_gauss_%.1f" % lda_factor, pac_gauss),
                    ("time", endtime - starttime),
                    ("pac_opt_%.1f" % lda_factor, pac_opt),
                    ("time", endtime_opt - starttime)]
    return returnv, theta_1d
