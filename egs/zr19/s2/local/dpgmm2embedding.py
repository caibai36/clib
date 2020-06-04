from typing import Optional

import os
import sys
import argparse
import logging
logging.basicConfig(stream=sys.stdout,level=logging.INFO, format="[ %(asctime)s | %(filename)s | %(levelname)s ] %(message)s", datefmt="%d/%m/%Y %H:%M:%S")

import numpy as np
from scipy.stats import chi2
from scipy.stats import multivariate_normal
# from sklearn import metrics

import kaldi_io
if not 'KALDI_ROOT' in os.environ:
    # Default! To change run python with 'export KALDI_ROOT=/some_dir python'
    os.environ['KALDI_ROOT'] = '/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi'

# https://en.wikipedia.org/wiki/Inverse-Wishart_distribution
class NormalInverseWishartDistribution(object):
    def __init__(self, mu, lmbda, nu, psi):
        self.mu = mu
        self.lmbda = float(lmbda)
        self.nu = nu
        self.psi = psi
        self.inv_psi = np.linalg.inv(psi)

    def sample(self):
        sigma = np.linalg.inv(self.wishartrand())
        return (np.random.multivariate_normal(self.mu, sigma / self.lmbda), sigma)

    def wishartrand(self):
        dim = self.inv_psi.shape[0]

        chol = np.linalg.cholesky(self.inv_psi)
        foo = np.zeros((dim,dim))

        for i in range(dim):
            for j in range(i+1):
                if i == j:
                    foo[i,j] = np.sqrt(chi2.rvs(self.nu-(i+1)+1))
                else:
                    foo[i,j]  = np.random.normal(0,1)
        return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))

    def posterior(self, data):
        n = len(data)
        mean_data = np.mean(data, axis=0)
        sum_squares = np.sum([np.array(np.matrix(x - mean_data).T * np.matrix(x - mean_data)) for x in data], axis=0)
        mu_n = (self.lmbda * self.mu + n * mean_data) / (self.lmbda + n)
        lmbda_n = self.lmbda + n
        nu_n = self.nu + n
        psi_n = self.psi + sum_squares + self.lmbda * n / float(self.lmbda + n) * np.array(np.matrix(mean_data - self.mu).T * np.matrix(mean_data - self.mu))
        return NormalInverseWishartDistribution(mu_n, lmbda_n, nu_n, psi_n)

def DPGMM3(alpha, mu0, lmbda, Sigma0, nu, sources, K0, num_iterations, verbose=True, targets=None, sources_test=None, targets_test=None):
    """ Implementation of the DPGMM clustering by the Gibb sampling.

    Parameters
    ----------
    alpha: the concentration parameter
    mu0: the prior belief of the mean (np.array with shape [data_dim])
    lmbda: the belief-strength of the mean
    Sigma0: the prior belief of the covariance (np.array with shape [data_dim, data_dim])
    nu: the belief-strength of the covariance (degree of freedom (df)); df>=dim_data+3
    sources: the data (np.array with shape [num_data, data_dim])
    K0: the initial number of the clusters (can be any number)
    num_iterations: number of the iterations
    verbose: print verbose information or not
    targets: for evaluation (np.array with shape [num_data_target])
    sources_test: train model at sources to get parameters, 
        use these parameters to get cluster indicators z, 
        posteriorgram z_posterior and clusters_test for the sources_test
        (np.array with shape [num_data_test, data_dim])
    targets_test: for evaluation of sources_test (np.array with shape [num_data_target])

    Returns
    -------
    NOTE: the num_clusters includes a possible new cluster.
    A dict of key-value:
        {'z': z, 'z_posterior': z_posterior, 'weights': weights, 'means': means, 'covs': covs, 'clusters': clusters}
    z: the sampled hidden cluster indicators of the last iteration (shape [num_data])
    z_posterior: the posterior of all the data (shape [num_data, num_clusters])
    weights: the sampled weights of the last iteration (shape [num_clusters])
    means: the sampled mean of the last iteration (list of np.array with length num_clusters)
    covs: the sampled variance of the last iteration (list of np.array with length num_clusters)
    clusters: the cluster index set of the last iteration (shape [num_clusters]) (not update for sources_test) (from z)
    clusters_test: the cluster index set for the cluster_test (number of clusters_test maybe less than clusters)
    
    Example
    -------
    import scipy
    from sklearn import cluster, datasets
    from itertools import cycle, islice

    n_samples = 1500
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    sources, targets = blobs[0], blobs[1]

    D = sources.shape[1]
    alpha = 1 # concentrate parameter
    mu0 = np.mean(sources, axis = 0)
    lmbda = 1 # belief of the mean
    Sigma0 = np.cov(sources.T) # belief of the covariance; for np.cov(m), each column of m is a single observation
    nu = D + 3 # belief of variance or degree of freedom
    K0=10 # initial number of clusters
    num_iterations = 100
    verbose=True

    results = DPGMM2(alpha, mu0, lmbda, Sigma0, nu, sources, K0, num_iterations, verbose)
    # preds = results['z']
    preds = np.argmax(results['z_posterior'], axis=1)

    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(preds) + 1))))
    plt.scatter(sources[:, 0], sources[:, 1], s=10, color=colors[preds])
    plt.show()

    implemented by bin-wu at 15:38 in 2020.02.09
    """
    # randomly initilize the cluster indicator
    z = np.random.choice(K0, len(sources))
    niw_sampler = NormalInverseWishartDistribution(mu0, lmbda, nu, Sigma0)

    for iteration in range(num_iterations):
        # sample the weights
        clusters, counts = np.unique(z, return_counts=True) # counter, remove empty clusters
        num_clusters = len(clusters)
        if verbose: logging.info(f"iter: {iteration} - num_clusters: {num_clusters}")
        clusters = np.append(clusters, [clusters.max() + 1]) # 1,....,K, K+1 --- not empty clusters + possible new cluster
        counts = np.append(counts, [alpha]) # n_1,.., n_K, n_{K+1}
        weights = np.random.dirichlet(counts)

        # sample the mean and the covariance for each Gaussian cluster including the possible new K+1 one
        parameters = []
        for cluster in clusters[:-1]: # exclude the new K+1 cluster
            parameters.append(niw_sampler.posterior(sources[z==cluster]).sample())
        parameters.append(niw_sampler.sample()) # sample for the K+1 cluster
        means, covs = list(zip(*parameters))

        # sample z
        num_clusters = len(clusters) # number of not empty clusters and possible new one
        num_sources = len(sources)
        assert (len(clusters) == len(means) == len(covs) == len(weights))

        z_posterior = []
        for k in range(num_clusters):
            z_posterior.append((multivariate_normal.pdf(sources, mean=means[k], cov=covs[k]) * weights[k]).reshape(-1, 1))
        z_posterior = np.concatenate(z_posterior, axis=1)
        z_posterior = z_posterior / z_posterior.sum(axis=1).reshape(-1, 1)

        updated_z = []
        for i in range(num_sources):
            updated_z.append(np.random.choice(clusters, p=z_posterior[i]))

        # update z
        z = np.array(updated_z)

        # evaluation
        if targets is not None:
            preds = np.argmax(z_posterior, axis=1) # preds = z
            result = metrics.homogeneity_completeness_v_measure(targets, preds)
            if verbose: logging.info('iter: {}, num_clusters: {}, homo: {:.4f}, comp: {:.4f}, v_meas: {:.4f}'.format(iteration, len(clusters), result[0], result[1], result[2]))
    
    clusters_test = None
    if sources_test is not None:
        z_posterior = []
        for k in range(num_clusters):
            z_posterior.append((multivariate_normal.pdf(sources_test, mean=means[k], cov=covs[k]) * weights[k]).reshape(-1, 1))
        z_posterior = np.concatenate(z_posterior, axis=1)
        z_posterior = z_posterior / z_posterior.sum(axis=1).reshape(-1, 1)
        
        num_sources_test = len(sources_test)
        updated_z = []
        for i in range(num_sources_test):
            updated_z.append(np.random.choice(clusters, p=z_posterior[i]))

        # update z
        z = np.array(updated_z)
        
        # update clusters_test
        clusters_test, counts = np.unique(z, return_counts=True) # counter, remove empty clusters
    
    if (sources_test is not None) and (targets_test is not None) and verbose:
        preds_test = np.argmax(z_posterior, axis=1) # preds = z
        result = metrics.homogeneity_completeness_v_measure(targets_test, preds_test)
        if verbose: logging.info('train_iter: {}, num_clusters_test: {}, homo: {:.4f}, comp: {:.4f}, v_meas: {:.4f}'.format(iteration, len(clusters_test), result[0], result[1], result[2]))

    return  {'z': z, 'z_posterior': z_posterior, 'weights': weights, 'means': means, 'covs': covs, 'clusters': clusters, 'clusters_test': clusters_test}

def indices2onehot_np(class_indices: np.array, num_classes: Optional[int] = None) -> np.array:
    """
    Convert a sequence of indices its onehot representation (numpy version)
    
    Example
    -------
    In [30]: %run local/dpgmm2embedding.py

    In [31]: indices2onehot_np(np.array([1, 0, 3]))
    Out[31]: 
    array([[0., 1., 0., 0.],
           [1., 0., 0., 0.],
           [0., 0., 0., 1.]])
    """
    num_classes = class_indices.max() + 1 if num_classes is None else num_classes
    onehot = np.zeros((class_indices.size, num_classes))
    onehot[np.arange(class_indices.size), class_indices] = 1
    return onehot

parser = argparse.ArgumentParser(description=f"Train DPGMM clustering on training features and test it on test feautures",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--feat_train", type=str, default="data/test3utts/feats.scp", help="the training feat scp file")
parser.add_argument("--feat_test", type=str, default="data/test3utts/feats.scp", help="the test feat scp file")
parser.add_argument("--result", type=str, default="eval/abx/embedding/exp/dpgmm/test3utt_mfcc13",
                    help="dir ${result}_onehot for the onehot representation and dir ${result}_prob for the posterior representation.")
parser.add_argument("--K0", type=int, default=10, help="the initial number of the clusters (can be any number)")
parser.add_argument("--num_iterations", type=int, default=100, help="number of the iterations")
parser.add_argument("--verbose", type=bool, default=True, help="print verbose information or not")

parser.add_argument("--seed", type=int, default=2020, help="seed for sampling")
parser.add_argument("--alpha", type=float, default=1, help="the concentration parameter")
parser.add_argument("--lmbda", type=float, default=1, help="the belief-strength of the mean")
parser.add_argument("--nu", type=float, default=None, help="the belief-strength of the covariance (degree of freedom (df)); df>=dim_data+3")

args = parser.parse_args()
np.random.seed(seed=args.seed)

train_features = []
for uttid, feature in kaldi_io.read_mat_scp(args.feat_train):
    train_features.append(feature)
sources = np.concatenate(train_features, axis=0)

test_features = []
test_lengths = []
uttids = []
for uttid, feature in kaldi_io.read_mat_scp(args.feat_test):
    test_features.append(feature)
    test_lengths.append(feature.shape[0])
    uttids.append(uttid)
sources_test = np.concatenate(test_features, axis=0)

D = sources.shape[1]
alpha = args.alpha # concentrate parameter
lmbda = args.lmbda # belief of mean
nu = D + 3 if args.nu is None else args.nu # belief of variance or degree of freedom
mu0 = np.mean(sources, axis = 0)
Sigma0 = np.cov(sources.T)

# initial number of clusters
K0 = args.K0
num_iterations = args.num_iterations
verbose=args.verbose
results=DPGMM3(alpha, mu0, lmbda, Sigma0, nu, sources, K0, num_iterations, verbose, 
               targets=None, sources_test=sources_test, targets_test=None)

####################################
prob = results['z_posterior']
indices = np.argmax(prob, axis=1)
onehot = indices2onehot_np(indices)

onehot_dir = args.result + "_onehot"
prob_dir = args.result + "_prob"

if not os.path.exists(prob_dir):
    os.makedirs(prob_dir)

if not os.path.exists(onehot_dir):
    os.makedirs(onehot_dir)

# Split the merged feature by the length of each utterance
start_index = 0
for i in range(len(test_lengths)):
    embedding_file = os.path.join(prob_dir, uttids[i] + ".txt")
    np.savetxt(embedding_file, prob[start_index:start_index+test_lengths[i]], fmt='%.6e')
    start_index += test_lengths[i]

start_index = 0
for i in range(len(test_lengths)):
    embedding_file = os.path.join(onehot_dir, uttids[i] + ".txt")
    np.savetxt(embedding_file, onehot[start_index:start_index+test_lengths[i]], fmt='%d')
    start_index += test_lengths[i]
