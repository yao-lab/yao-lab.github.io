import pandas as pd
import numpy as np
from scipy.stats import norm

class hodge_rank:
    def __init__(self, data):
        df = pd.read_csv(data)
        m = max(df['Voter'])
        n = np.amax(df[['Choice1', 'Choice2']])
        self.m = m
        self.n = n
        
        X, y = [], []
        idx = df['Voter']
        tmp = [[] for _ in range(m)]
        W = [np.zeros((n, n)) for _ in range(m)]
        Y = [np.zeros((n, n)) for _ in range(m)]
        for (i, j) in enumerate(idx):
            tmp[j - 1].append(i)
        for j in range(m):
            X.append( df[['Choice1', 'Choice2']].values[tmp[j]] )
            y.append( df['Result'].values[tmp[j]] )
            for k in range(len(tmp[j])):
                Y[j][ X[j][k, 0] - 1, X[j][k, 1] - 1 ] += y[j][k]
                Y[j][ X[j][k, 1] - 1, X[j][k, 0] - 1 ] += -y[j][k]
                W[j][ X[j][k, 0] - 1, X[j][k, 1] - 1 ] += 1
                W[j][ X[j][k, 1] - 1, X[j][k, 0] - 1 ] += 1
            non_zero_indices = np.nonzero(W[j])
            Y[j][non_zero_indices] = Y[j][non_zero_indices] / W[j][non_zero_indices]
        
        self.data = [X, y, W, Y]
        
    def train(self, model = 'Uniform', curl_proj = False, ep = 1e-5):
        assert model in ['Uniform', 'Bradley-Terry', 'Thurstone-Mosteller', 'Angular transform']
        
        n = self.n
        m = self.m
        W_0 = np.zeros((n, n))
        Y_0 = np.zeros((n, n))
        Pi = np.ones((n, n)) / 2
        for i in range(m):
            W_0 += self.data[2][i]
            Y_0 += self.data[2][i] * self.data[3][i]
        non_zero_indices = np.nonzero(W_0)
        Pi[non_zero_indices] = ( Y_0[non_zero_indices] / W_0[non_zero_indices] + np.ones((n, n))[non_zero_indices] ) / 2
        ne_inf_indices = (Pi < ep)
        po_inf_indices = (Pi > (1 - ep) )
        self.W = W_0
        
        if model == 'Uniform':
            self.Y = 2 * Pi - np.ones((n, n))
        if model == 'Bradley-Terry':
            Pi[ne_inf_indices] = ep
            Pi[po_inf_indices] = 1 - ep
            self.Y = np.log( Pi / (np.ones((n, n)) - Pi) )
        if model == 'Thurstone-Mosteller':
            Pi[ne_inf_indices] = ep
            Pi[po_inf_indices] = 1 - ep
            self.Y = norm.ppf(Pi)
        if model == 'Angular transform':
            self.Y = np.arcsin( 2 * Pi - np.ones((n, n)) )
            
        self.sparse_data = get_sparse(W_0, self.Y)
        [self.score, self.global_component] = global_score(self.sparse_data)
        self.ranking = np.argsort( np.argsort( -self.score.reshape(-1) ) ) + 1
        
        numE = len( self.sparse_data[1] )
        GG = self.sparse_data[1].reshape( (numE, 1) )
        yy = self.sparse_data[2].reshape( (numE, 1) )
        self.R = yy - self.global_component
        self.Cp = np.sum( GG * (self.R ** 2) ) / np.sum( GG * (yy ** 2) )
        if curl_proj:
            [self.Phi, self.curl_component] = curl(self.sparse_data)
            self.Inc_curl = np.sum( GG * ( self.curl_component ** 2 ) ) / np.sum( GG * (yy ** 2) )
            self.Inc_harm = self.Cp - self.Inc_curl
            
        tri = self.sparse_data[4]
        Y = self.Y
        score = self.score.reshape(-1)
        cr = np.zeros(len(tri))
        for k,t in enumerate(tri):
            i = t[0]
            j = t[1]
            l = t[2]
            rou = np.abs( Y[i, j] + Y[j, l] + Y[l, i] )
            a = rou / np.abs( 3 * ( score[i] - score[j] ) )
            b = rou / np.abs( 3 * ( score[j] - score[l] ) )
            c = rou / np.abs( 3 * ( score[l] - score[i] ) )
            cr[k] = np.mean( [a, b, c] )
        self.cr = cr
        
    def train_individual(self):
        W = self.data[2]
        Y = self.data[3]
        n = self.n
        m = self.m
        
        Cp_Individual = np.zeros(m)
        for k in range(m):
            data_k = get_sparse(W[k], Y[k])
            [score_k, global_component_k] = global_score(data_k)
            numE_k = len( data_k[1] )
            GG_k = data_k[1].reshape( (numE_k, 1) )
            yy_k = data_k[2].reshape( (numE_k, 1) )
            R_k = yy_k - global_component_k
            Cp_Individual[k] = np.sum( GG_k * (R_k ** 2) ) / np.sum( GG_k * (yy_k ** 2) )
        
        self.Cp_Individual = Cp_Individual
        
####################################################################################################################################################################################################################################################################################################################################################

def get_sparse(W, Y):
    num = int(len( np.nonzero(W)[0] )/2)
    n = len(W)
    GG = np.zeros(num)
    y = np.zeros(num)
    edge, triangle = [], []
    k = 0
    for j in range(n - 1):
        for i in range(j + 1, n):
            if W[i, j] > 0:
                GG[k] = W[i, j]
                y[k] = Y[i, j]
                k += 1
                edge.append( (i,j) )
            if i < n - 1:
                for q in range(i + 1, n):
                    if W[j, i] > 0 and W[i, q] > 0 and W[q, j] > 0:
                        triangle.append( (j,i,q) )
                
    return [n, GG, y, edge, triangle]

def global_score(sparse_data):
    n = sparse_data[0]
    GG = sparse_data[1]
    y = sparse_data[2]
    edge = sparse_data[3]
    k = len(GG)
    d0 = np.zeros((k,n))
    for i in range(k):
        d0[i, edge[i][0]] = 1
        d0[i, edge[i][1]] = -1
    d0_star = d0.T @ np.diag(GG)
    L0 = d0_star @ d0
    Div = d0_star @ y.reshape((k,1))
    theta = np.linalg.lstsq(L0, Div, rcond=1e-5)[0]
    
    return [theta, d0 @ theta]

def curl(sparse_data):
    n = sparse_data[0]
    GG = sparse_data[1]
    y = sparse_data[2]
    edge = sparse_data[3]
    triangle = sparse_data[4]
    numE = len(GG)
    numT = len(triangle)
    d1 = np.zeros((numT,numE))
    for k in range(numT):
        a = triangle[k][1:2] + triangle[k][:1]
        b = triangle[k][2:] + triangle[k][1:2]
        c = triangle[k][2:] + triangle[k][:1]
        for m in range(numE):
            if edge[m] == a:
                d1[k, m] = 1
            if edge[m] == b:
                d1[k, m] = 1
            if edge[m] == c:
                d1[k, m] = -1
        
    d1_star = np.diag( 1 / GG ) @ d1.T
    L1 = d1 @ d1_star
    Curl = d1 @ y.reshape((numE,1))
    Phi = np.linalg.lstsq(L1, Curl, rcond=1e-5)[0]
    
    
    return [Phi, d1_star @ Phi]