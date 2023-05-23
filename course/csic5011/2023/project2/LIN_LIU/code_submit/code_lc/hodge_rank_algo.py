import numpy as np
import scipy
import scipy.sparse as scispa
from scipy.linalg import pinvh
import copy
import cvxpy as cp
from knockpy import KnockoffFilter
class hodge_rank:
    def __init__(self, data,N,M, n):
        self.data = data
        self.N = N
        self.M = M
        self.n = n
        self.edge_set = []
        for k in range(N):
            if (self.data[k][1], self.data[k][2]) not in self.edge_set:
                self.edge_set.append((self.data[k][1], self.data[k][2]))
        self.E = len(self.edge_set)
       
        # self.edge_set.sort()
        # print(self.edge_set)
        # ssss
        self.edge_set_dict = {}
        for k in range(len(self.edge_set)):
            self.edge_set_dict[self.edge_set[k]] = k
        self.row = [i[0] for i in self.edge_set]
        self.col = [i[1] for i in self.edge_set]

    def get_delta_matrix(self):
        delta_matrix = np.zeros((self.E,self.n))
        for k in range(len(self.row)):
            delta_matrix[k, self.row[k]] = 1
            delta_matrix[k, self.col[k]] = -1
        return delta_matrix


    def get_graph(self , filter=None):
        ## input : Nx4 N: number of pairs 4 : annotator,left, right, preference -1 0 1
        ## output : matrix with E
        self.value_matrix = np.zeros((self.E,))
        self.count_matrix = np.zeros((self.E,))
        self.norm_matrix = np.zeros((self.E,))
        
        for k in range(self.N):
            self.value_matrix[self.edge_set_dict[(self.data[k][1], self.data[k][2])]] += 1
            self.norm_matrix[self.edge_set_dict[(self.data[k][1], self.data[k][2])]] += 1
            if (self.data[k][2], self.data[k][1]) in self.edge_set_dict.keys():
                self.norm_matrix[self.edge_set_dict[(self.data[k][2], self.data[k][1])]] += 1
                self.value_matrix[self.edge_set_dict[(self.data[k][2], self.data[k][1])]] -= 1
            self.count_matrix[self.edge_set_dict[(self.data[k][1], self.data[k][2])]] += 1
        self.value_matrix = self.value_matrix/ (self.norm_matrix)


    def get_annotator_matrix(self):
        ## input : Nx4 N: number of pairs 4 : annotator,left, right, preference -1 0 1
        ## output : MxE : M number of annotator, E number of edges.
        annotator_matrix = np.zeros(( self.E, self.M))
        for k in range(self.N):
            annotator_matrix[self.edge_set_dict[(self.data[k][1], self.data[k][2])],self.data[k][0]]=1
            # if (self.data[k][2], self.data[k][1]) in self.edge_set_dict.keys():
            #     annotator_matrix[self.edge_set_dict[(self.data[k][1], self.data[k][2])],self.data[k][0]]=1
        return annotator_matrix

    def get_global_rank(self):
        ## Y E  Y= delta \theta +\epsilon  delta : Exn
        ## 1/2  ||Y- D \theta||_W^2
        ## theta = (WDTD)-1 WDTY
        D = self.get_delta_matrix()
        W = np.diag(self.count_matrix)
        Y = self.value_matrix
        Mat1 = np.matmul(np.matmul(D.transpose(), W), D)
        Mat2 = np.matmul(np.matmul(D.transpose(),W), Y)
        np.save('mat1.npy', Mat1)
        np.save('mat2.npy', Mat2)
        # print(np.diag(Mat1)[:5])
        theta = np.linalg.lstsq(Mat1, Mat2, rcond=1e-5)[0]
        return theta
    
    # def get_outlier_lbi(self, kappa=10, alpha=1, max_iter=100000, lam=1):
    #     D = self.get_delta_matrix()
    #     # W = np.diag(self.count_matrix)
    #     # W = np.eye(len(self.count_matrix))
    #     Y = self.value_matrix
    #     Mat1 = np.matmul(np.matmul(D.transpose(), W), D)
    #     Mat2 = np.matmul(np.matmul(D.transpose(),W), Y)
    #     theta = np.matmul(np.linalg.pinv(Mat1), Mat2)
    #     #### begin itertaion
    #     w = np.zeros((self.M,))
    #     gamma = np.zeros((self.M,))
    #     A = self.get_annotator_matrix()
    #     gamma_path = [copy.deepcopy(gamma)]
    #     print(alpha)
    #     t_list = [0]
    #     rho_list = [np.zeros((self.M,))]
    #     S_list = [[],]
    #     for i in range(max_iter):
            
    #         # grad = np.matmul(A.transpose(), res)
    #         res = (Y - np.matmul(D, theta)-np.matmul(A, gamma))
    #         th_grad = np.matmul(D.transpose(), res)
    #         grad = np.matmul(A.transpose(), res)
    #         ## find t
    #         t_new = self.update_t(grad, t_list[i],rho_list[i])
    #         t_list.append(t_new)
    #         rho_new = rho_list[i]+ (t_new-t_list[i]) * grad 
    #         rho_list.append(copy.deepcopy(rho_new))
    #         # print(rho_new)
    #         S_k = [i for i in range(len(rho_new)) if rho_new[i]>=0.99]
    #         print(S_k)
    #         gamma_tmp = cp.Variable(self.M)
    #         cost = cp.sum_squares(Y - np.matmul(D, theta)-A@gamma_tmp)
    #         selet_vector = np.zeros((self.M,))
    #         selet_vector[S_k]=1
    #         selet_vector_2 = np.ones((self.M,))
    #         selet_vector_2[S_k]=0
            
    #         cs = [np.diag(selet_vector)@gamma_tmp>=0, np.diag(selet_vector_2)@gamma_tmp==0]
    #         prob = cp.Problem(cp.Minimize(cost),cs)
    #         # prob.solve()
    #         # theta = np.linalg.lstsq(Mat1, np.matmul(np.matmul(D.transpose(),W), Y-np.matmul(A, gamma)), rcond=1e-5)[0]
    #         # theta = theta + kappa * alpha * th_grad
    #         gamma= gamma_tmp.value
    #         gamma_path.append(copy.deepcopy(gamma))
    #     return theta, gamma, np.array(gamma_path)
    
    # def update_t(self,grad, t, rho):
    #     t_new = cp.Variable(1)
    #     cost = -t_new
    #     cs = [t_new>=t, rho+ (t_new-t)  * grad <= 1, rho- (t_new-t)  * grad >= -1]
    #     prob = cp.Problem(cp.Minimize(cost),cs)
    #     prob.solve()
    #     # print(t_new.value)
    #     return t_new.value
        
        
    

    def get_outlier_lbi(self, kappa=5, alpha= 5e-3, max_iter=10000, lam=1):
        D = self.get_delta_matrix()
        W = np.diag(self.count_matrix)
        # W = np.eye(len(self.count_matrix))
        Y = self.value_matrix
        Y = (Y - np.mean(Y))/np.std(Y)
        Mat1 = np.matmul(np.matmul(D.transpose(), W), D)
        Mat2 = np.matmul(np.matmul(D.transpose(),W), Y)
        theta = np.matmul(np.linalg.pinv(Mat1), Mat2)
        #### begin itertaion
        w = np.zeros((self.M,))
        gamma = np.zeros((self.M,))
        A = self.get_annotator_matrix()
        A = A-np.mean(A,axis=0,keepdims=True)
        Cov = np.linalg.pinv(np.matmul(A.transpose(),A))
        A = np.matmul(A , Cov)
        gamma_path = [copy.deepcopy(gamma)]
        print(alpha)
        for i in range(max_iter):
            res = (Y - np.matmul(D, theta)-np.matmul(A, gamma))
            grad = np.matmul(A.transpose(), res)
            th_grad = np.matmul(D.transpose(), res)
            # if np.linalg.norm(grad, 2) <= tol:
            #     break
            w = w + alpha*grad
            gamma = kappa * self.shrikage(w, lam)
            theta = theta + kappa * alpha * th_grad
            gamma_path.append(copy.deepcopy(gamma))
        return theta, gamma, np.array(gamma_path)

    def get_outlier_lbi_with_knockoffs(self, kappa=10 , alpha=0.00001, max_iter=100000, lam=1, q=0.1):
        D = self.get_delta_matrix()
        W = np.diag(self.norm_matrix)
        Y = self.value_matrix
        Y = (Y - np.mean(Y))/np.std(Y)
        Mat1 = np.matmul(np.matmul(D.transpose(), W), D)
        Mat2 = np.matmul(np.matmul(D.transpose(),W), Y)
        theta = np.matmul(np.linalg.pinv(Mat1), Mat2)
        #### begin itertaion
        w = np.zeros((2*self.M,))
        gamma = np.zeros((2 * self.M,))
        # A = self.get_annotator_matrix()
        A = self.get_annotator_matrix()
        A = A-np.mean(A,axis=0,keepdims=True)
        Cov = np.linalg.pinv(np.matmul(A.transpose(),A))
        A = np.matmul(A , Cov)
        ### gouzao A *
        H = np.matmul(np.matmul(D, np.linalg.pinv(np.matmul(D.transpose(),D),rcond=1e-3)), D.transpose())
        H_new = np.eye(len(H)) -H
        Y_kn = np.matmul(H_new, Y)
        X_kn = np.matmul(H_new, A)
    
        # kfilter1 = KnockoffFilter(ksampler='gaussian', knockoff_kwargs={'method':'maxent'})
        
        # kfilter1.forward(X_kn,Y_kn )
        # xk = kfilter1.Xk
        # np.save('college_Xk.npy',kfilter1.Xk)
        xk = np.load('college_Xk.npy',)
       
        
        A_tilde = np.matmul(np.linalg.pinv(H_new, rcond=1e-5), xk)
        
        print(A_tilde.shape)
        A_whole =  np.concatenate([A, A_tilde],1)
        print(A_whole.shape)

        gamma_path = [copy.deepcopy(gamma)]
        print(alpha)
        the_list = []
        first_time = np.zeros((2 * self.M))
        first_time_path = []
        for i in range(max_iter):
            res = (Y - np.matmul(D, theta)-np.matmul(A_whole, gamma))
            grad = np.matmul(A_whole.transpose(), res)
            th_grad = np.matmul(D.transpose(), res)
            w = w + alpha*grad
            gamma = kappa * self.shrikage(w, lam)
            theta = theta + kappa * alpha * th_grad
            ### update
            new_t = [k for k in range(2*self.M) if gamma[k] !=0 and gamma_path[-1][ k]==0]
            if len(new_t) !=0:
                the_list.append(1/(i+1))
                first_time[new_t] = 1/(1+i)
                gamma_path.append(copy.deepcopy(gamma))
                first_time_path.append(copy.deepcopy(first_time))
        first_time_path = np.array(first_time_path)
        stop_k = len(the_list)
        for  k, th in enumerate(the_list):
            if k==0:
                continue
            cur_rt = np.sum(first_time_path[:k], axis=0)
            W_st = np.sign(cur_rt[ :self.M] - cur_rt[ self.M:]) * np.abs(cur_rt[ :self.M])
            index_1 = [i for i in range(len(W_st)) if W_st[i] < 0] 
            index_2 = [i for i in range(len(W_st)) if cur_rt[i] > 0] 
            fdr_pre = ((len(index_1))) / max(1, len(index_2))
            print(th, fdr_pre)
           
        # print(fdr_pre)
            if fdr_pre <= q:
                stop_k = k
        
        gamma_path = gamma_path[:stop_k+1]
        gamma_path = np.array(gamma_path)[:, :self.M]
        print(gamma_path.shape)
        
        gamma= gamma_path[stop_k]
        # cur_rt = np.sum(first_time_path[:stop_k], axis=0)
        import pandas as pd
        
        x = pd.read_csv('college_prefer.csv')
        print(x['ratio'].iloc[gamma!=0])
        
        return theta, gamma, np.array(gamma_path), 
    
    
        
    def shrikage(self, x, lam=1):
        return np.sign(x) * np.clip(np.abs(x)-lam,a_min=0, a_max=1)
