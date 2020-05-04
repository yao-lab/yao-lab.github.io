import os

import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import bisect

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal, Normal, Cauchy, Chi2, StudentT, Uniform
import torch.autograd as autograd

from network import *
from utils import kendall

# import matplotlib.pyplot as plt

class fgan():
    def __init__(self, p, eps, device=None, epsilon=1e-5):
        self.p = p
        self.eps = eps
        self.epsilon = epsilon
        self.device = device if device is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def dist_init(self, true_type='Gaussian', cont_type='Gaussian', t_df=None, c_df=None,
                  t_cov_type='spherical', t_loc=0, t_cov=1,
                  c_cov_type='spherical', c_loc=0, c_cov=None, delta=None):
        ## true type: Cauchy/Student/Gaussian
        ## Cont type: Cauchy/Student/Gaussian/Uniform/Delta
        
        self.true_type = true_type
        self.cont_type = cont_type
        self.t_df = t_df
        self.c_df = c_df
        self.delta = delta
        if self.true_type == 'Cauchy':
            self.true_type = 'Student'
            self.t_df = 1
        if self.cont_type == 'Cauchy':
            self.cont_type = 'Student'
            self.c_df = 1

       ## true sampler
        self.t_cov_type = t_cov_type
        self.t_loc = torch.zeros(self.p) + t_loc

        if self.t_cov_type == 'spherical':
            if not isinstance(float(t_cov), float):
                raise ValueError('Need a float number for spherical t_cov')
            self.t_cov = torch.eye(self.p) * t_cov

        elif self.t_cov_type == 'full':
            if not isinstance(t_cov, np.ndarray):
                raise ValueError('Need a np array for full t_cov')
            self.t_cov = torch.from_numpy(t_cov).float()
        else:
            raise NameError('t_cov_type only has spherical and full')

        if self.true_type == 'Gaussian':
            self.t_d = MultivariateNormal(torch.zeros(self.p), covariance_matrix=self.t_cov)
        elif self.true_type == 'Student':
            assert self.t_df is not None, 'Specify degree of freedom for Student t distribution'
            self.t_normal_d = MultivariateNormal(torch.zeros(self.p), covariance_matrix=self.t_cov)
            self.t_chi2_d = Chi2(df=self.t_df)
        else:
            raise NameError('True type must be Gaussian/Student/Cauchy!')


        ## contamination sampler
        self.c_cov_type = c_cov_type
        self.c_loc = c_loc

        if self.cont_type in ['Gaussian', 'Student']:
            if self.c_cov_type == 'spherical':
                if not isinstance(float(c_cov), float):
                    raise ValueError('Need a float number for spherical c_cov')
                self.c_cov = torch.eye(self.p) * c_cov

            elif self.c_cov_type == 'full':
                if not isinstance(c_cov, np.ndarray):
                    raise ValueError('Need a np array for full c_cov')
                self.c_cov = torch.from_numpy(c_cov).float()

            else:
                raise NameError('c_cov_type only has spherical and full')


        if self.cont_type == 'Gaussian':
            self.c_normal_d = MultivariateNormal(torch.zeros(self.p), covariance_matrix=self.c_cov)
        elif self.cont_type == 'Student':
            assert self.c_df is not None, 'Specify degree of freedom for Student t distribution'
            self.c_normal_d = MultivariateNormal(torch.zeros(self.p), covariance_matrix=self.c_cov)
            self.c_chi2_d = Chi2(df=self.c_df)
        elif self.cont_type == 'Uniform':
            self.c_uniform_d = Uniform(-5, 5)
        elif self.cont_type == 'Delta':
            pass
        else:
            raise NameError('Contamination type must be Gausssian/Student/Cauchy/Uniform/Delta')


    def data_init(self, train_size=50000, batch_size=100, data=None):
        # data must be torch tensor
        if data is None:
            self.Xtr = self._sampler(train_size)
            self.load_data = False
        else:
            self.Xtr = data
            self.load_data = True
        self.batch_size = batch_size
        self.poolset = PoolSet(self.Xtr)
        self.dataloader = DataLoader(self.poolset, batch_size=self.batch_size, shuffle=True)

    def _sampler(self, n):
        ## true sampler     
        if self.true_type == 'Gaussian':
            t_x = self.t_d.sample((n, )) + self.t_loc
        elif self.true_type == 'Student':
            t_normal_x = self.t_normal_d.sample((n, )) # [n, p]
            t_chi2_x = self.t_chi2_d.sample((n,))
            t_x = t_normal_x / (torch.sqrt(t_chi2_x.view(-1, 1)/self.t_df) + self.epsilon)
            t_x = t_x + self.t_loc
        else:
            raise NameError('True type must be Gaussian/Student/Cauchy')  

        ## contamination sampler
        if self.cont_type == 'Gaussian':
            c_x = self.c_normal_d.sample((n, )) + self.c_loc
        elif self.cont_type == 'Student':
            c_normal_x = self.c_normal_d.sample((n, )) # [n, p]
            c_chi2_x = self.c_chi2_d.sample((n,))
            c_x = c_normal_x / (torch.sqrt(c_chi2_x.view(-1, 1)/self.c_df) + self.epsilon)
            c_x = c_x + self.c_loc
        elif self.cont_type == 'Uniform':
            c_x = self.c_uniform_d.sample((n, self.p))
        elif self.cont_type == 'Delta':
            c_x = torch.ones(n, self.p) * self.delta
        else:
            raise NameError('Contamination type must be Gausssian/Student/Cauchy/Uniform/Delta')

        s = (torch.rand(n) < self.eps).float()
        x = (t_x.transpose(1,0) * (1-s) + c_x.transpose(1,0) * s).transpose(1,0)

        return x

    def _error(self, weight_est): ## \| W'W - I_p \|_op
        cov_est = torch.mm(weight_est.detach().transpose(1,0), weight_est.detach()) ## cuda
        err_op = np.linalg.norm(cov_est.cpu().numpy() - self.t_cov.numpy(), ord=2)
        err_fr = (cov_est.to(self.device) - self.t_cov.to(self.device)).norm(2).item()
        return err_op, err_fr

    def _reweight(self, N=100000):
        if not hasattr(self, 'epr'): ## expect value of R
            if self.true_type == 'Student':
                t_student = StudentT(df=self.t_df)
                x = t_student.sample((5000000,))
            elif self.true_type == 'Gaussian':
                t_normal = Normal(0, 1)
                x = t_normal.sample((5000000,))
            self.epr = self._Rfunc(x).mean().item()

        def sov_func(a):
            r = 0
            rounds = N//500
            for _ in range(rounds): 
                if self.inverse_gaussian:
                    _xi1 = torch.randn(500, self.input_dim_G//2).to(self.device)
                    _xi2 = torch.randn(500, self.input_dim_G//2).to(self.device)
                    _xi2.data = 1/(torch.abs(_xi2.data) + self.epsilon)
                    xi = self.netGXi(torch.cat([_xi1, _xi2], dim=1)).view(500, -1).detach()
                else:
                    _xi = torch.randn(500, self.input_dim_G).to(self.device)
                    xi = self.netGXi(_xi.to(self.device)).view(500, ).detach() #(500, )

                _z = torch.randn(500, self.p).to(self.device)
                _vu = (_z[:, 0].div_(_z.norm(2, dim=1) + self.epsilon)).to(self.device) #(500, )
                
                r += self._Rfunc(a * xi * _vu).mean().item()

            return (r/rounds - self.epr)

        if sov_func(250) > 0:
            down = 0; up = 300
        else:
            print('Larger than 250!!!!!!!!!!')
            return 250

        factor = bisect(sov_func, down, up)
        return factor


    def _Rfunc(self, x, mode='ramp'):
        if mode == 'abs':
            return torch.abs(x)
        elif mode == 'ramp':
            return F.hardtanh(torch.abs(x))   


    def net_init(self, hidden_units, 
             activation_D='Sigmoid', activation_D1='Ramp', activation_Dn = 'ReLU',
             elliptical=True, input_dim_G=10, activation_G='LeakyReLU', hidden_units_G=[10, 10],
             init_G = 'kendall', subsample=None, init_D1=None,  init_D = 'xavier',
             use_bias=False, prob=False):

        self.use_bias = use_bias
        self.elliptical = elliptical
        self.input_dim_G = input_dim_G
        self.prob = prob

        if self.elliptical:
            assert (input_dim_G % 2 == 0), 'input_dim_G should be an even number'
            self.netGXi = GeneratorXi(activation=activation_G, 
                        input_dim=input_dim_G, hidden_units=hidden_units_G).to(self.device)
        self.netD = Discriminator(self.p, hidden_units, activation_D, 
                        activation_1=activation_D1, activation_n=activation_Dn, prob=self.prob).to(self.device)
        self.netG = Generator(self.p, elliptical=self.elliptical, 
                        use_bias=self.use_bias).to(self.device)
            
        ## initialize G's weight, prune 50% data (doesn't make sense, for example eps=0)
        if init_G == 'diag':
            Xmed = torch.median(self.Xtr, dim=0)[0]
            X2med = torch.median(self.Xtr**2, dim=0)[0]
            cov_est = torch.diag(X2med - Xmed**2)
        elif init_G == 'kendall':
            cov_est = kendall(self.Xtr, true=self.true_type, df=self.t_df, subsample=subsample)
        elif init_G == 'random':
            weight_est = torch.eye(self.p) + .1 * torch.randn(self.p, self.p)
            cov_est = weight_est.mm(weight_est.transpose(1,0))
        elif init_G == 'truth':
            cov_est = torch.eye(self.p)
        else:
            raise NameError('Initialization for G must be diag/kendall/random/truth')     
        # if init_G == 'cholesky':
        #     weight_est = np.linalg.cholesky(cov_est.numpy()).transpose(1,0) ## cov = LL^T = W^TW with W=L^T
        u, s, vt = np.linalg.svd(cov_est)
        weight_est = np.matmul(np.diag(s)**(1/2), vt) ## cov = USU^T, W = S^(1/2)U^T
        self.netG.weight.data.copy_(torch.from_numpy(weight_est).float().to(self.device))

        ## initialize G's bias if needed
        if self.use_bias:
            self.netG.bias.data.copy_(torch.median(self.Xtr, dim=0)[0].to(self.device))

        ## others' initialization
        if init_D in ['xavier', 'kaiming', 'normal']:
            self.netD.apply(eval('weights_init_' + init_D))
            if (self.elliptical):
                self.netGXi.apply(eval('weights_init_' + init_D))
            if init_D1 is not None:
                self.netD.feature.lyr1.weight.data.normal_(0, init_D1)
        else:
            raise NameError('Wrong Initialization')

    def optimizer_init(self, lr_d, lr_g, d_steps, g_steps=1, lr_gxi=None, 
                       decay_D=1.0, decay_G=1.0, sch_D=80, sch_G=50,
                       lam_g=0., lam_d=0., match='median', momentum=False, adam=False):
        ## match : median/mean.
        momentum = .1 if momentum is True else 0.
        
        if adam:
            self.optG = optim.Adam(self.netG.parameters(), lr=lr_g)
            self.optD = optim.Adam(self.netD.parameters(), lr=lr_d)
        else:
            self.optG = optim.SGD(self.netG.parameters(), lr=lr_g, momentum=momentum)
            if self.elliptical:
                if lr_gxi is None:
                    lr_gxi = lr_g
                self.optGXi = optim.SGD(self.netGXi.parameters(), lr=lr_gxi, momentum=momentum)
            self.optD = optim.SGD(self.netD.parameters(), lr=lr_d, momentum=momentum)
        self.g_steps = g_steps    
        self.d_steps = d_steps
        self.lam_g = lam_g
        self.lam_d = lam_d
        self.match = match
        if min(decay_G, decay_D) < 1.0:
            self.scheduler_D = optim.lr_scheduler.StepLR(self.optD, step_size=sch_D, gamma=decay_D)
            self.scheduler_G = optim.lr_scheduler.StepLR(self.optG, step_size=sch_G, gamma=decay_G)
            self.decay = True
        else:
            self.decay = False

    def fit(self, floss='js',
            epochs=100, avg_epochs=20, logd_trick=False,
            inverse_gaussian=True,
            tuning_check=None, thresh_cov=None, thresh_mean=None,
            verbose=10, show=True):
        ## after self.dist_init(), self.data_init(), net_init() and optimizer_init()
        ## gp_factor * Expect [(\|dD(x+delta)\|_2 - k) ** 2]

        if floss == 'js':
            assert (not self.prob), 'When using Logit loss, don\'t put Sigmoid at the last layer'
            criterion = nn.BCEWithLogitsLoss()
        elif floss == 'ls':
            assert self.prob, 'When using mse loss, sigmoid should be at the last layer'
            criterion = nn.MSELoss()
        else:
            raise NameError('floss is not defined')
        
        self.inverse_gaussian = inverse_gaussian

        self.loss_D = [] # record loss per epoch
        self.loss_G = []

        self.cov_est_record = []
        if self.use_bias:
            self.mean_est_record = []
        self.midweight = [] ## product of middle weight (Disc)
        self.lastweight = [] ## last weight (Disc)
        current_d_step = 1

        z_b = torch.zeros(self.batch_size, self.p).to(self.device)
        one_b = torch.ones(self.batch_size).to(self.device)
        if self.elliptical:
            if inverse_gaussian:
                xi_b1 = torch.zeros(self.batch_size, self.input_dim_G//2).to(self.device)
                xi_b2 = torch.zeros(self.batch_size, self.input_dim_G//2).to(self.device)
            else:
                xi_b = torch.zeros(self.batch_size, self.input_dim_G).to(self.device)

        for ep in range(epochs):
            ## update D
            if self.decay:
                self.scheduler_D.step()
                self.scheduler_G.step()
            loss_D_ep = []
            loss_G_ep = []
            loss_feat_D_ep = []
            loss_feat_G_ep = []
            loss_gp_ep = []  
            for ii, data in enumerate(self.dataloader):

                ## update D
                if data.shape[0] != self.batch_size:
                    continue
                self.netD.train()
                self.netD.zero_grad()
                x_real = data.to(self.device)
                feat_real, d_real_score = self.netD(x_real) ## (N, )
                if floss == 'js':
                    d_real_loss = criterion(d_real_score, one_b)
                elif floss == 'ls':
                    d_real_loss = criterion(d_real_score, one_b)

                if not self.elliptical:
                    if self.true_type == 'Student':
                        z_b.normal_() # [N, p]
                        z_b.data.div_(torch.sqrt(self.t_chi2_d.sample((self.batch_size, 1))/self.t_df).to(self.device) + self.epsilon)
                        x_fake = self.netG(z_b).detach()
                    elif self.true_type == 'Gaussian':
                        x_fake = self.netG(z_b.normal_()).detach()
                    else:
                        raise NameError('True type is not defined')

                else:
                    z_b.normal_()
                    z_b.div_(z_b.norm(2, dim=1).view(-1, 1) + self.epsilon)
                    if inverse_gaussian:
                        xi_b1.normal_()
                        xi_b2.normal_()
                        xi_b2.data = 1/(torch.abs(xi_b2.data) + self.epsilon)
                        xi = self.netGXi(torch.cat([xi_b1, xi_b2], dim=1)).view(self.batch_size, -1)
                    else:
                        xi_b.normal_()
                        xi = self.netGXi(xi_b).view(self.batch_size, -1)
                    x_fake = self.netG(z_b, xi).detach()

                feat_fake, d_fake_score = self.netD(x_fake)
                d_fake_loss = criterion(d_fake_score, 1-one_b)
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                loss_D_ep.append(d_loss.cpu().item())
                self.optD.step()
                    
                if current_d_step < self.d_steps:
                    current_d_step += 1
                    continue
                else:
                    current_d_step = 1

                ## update G
                self.netD.eval()
                for _ in range(self.g_steps):
                    self.netG.zero_grad()
                    if self.elliptical:
                        self.netGXi.zero_grad()

                    if not self.elliptical:
                        if self.true_type == 'Student':
                            z_b.normal_() # [N, p]
                            z_b.data.div_(torch.sqrt(self.t_chi2_d.sample((self.batch_size, 1))/self.t_df).to(self.device) + self.epsilon)
                            x_fake = self.netG(z_b)
                        elif self.true_type == 'Gaussian':
                            x_fake = self.netG(z_b.normal_())
                        else:
                            raise NameError('True type is not defined')

                    else:
                        z_b.normal_()
                        z_b.div_(z_b.norm(2, dim=1).view(-1, 1) + self.epsilon)
                        if inverse_gaussian:
                            xi_b1.normal_()
                            xi_b2.normal_()
                            xi_b2.data = 1/(torch.abs(xi_b2.data) + self.epsilon)
                            xi = self.netGXi(torch.cat([xi_b1, xi_b2], dim=1)).view(self.batch_size, -1)
                        else:
                            xi_b.normal_()
                            xi = self.netGXi(xi_b).view(self.batch_size, -1)
                        x_fake = self.netG(z_b, xi)

                    feat_fake, g_fake_score = self.netD(x_fake)
                    if (floss == 'js') & logd_trick:
                        g_fake_loss = criterion(g_fake_score, one_b)
                    elif (floss == 'js') or (floss == 'ls'):
                            g_fake_loss = - criterion(g_fake_score, 1-one_b)

                    g_fake_loss.backward()
                    loss_G_ep.append(g_fake_loss.cpu().item())
                    self.optG.step()
                    if self.elliptical:
                        self.optGXi.step()

            self.loss_D.append(np.mean(loss_D_ep))
            self.loss_G.append(np.mean(loss_G_ep))

            if (ep >= (epochs - avg_epochs)):
                self.cov_est_record.append(self.netG.weight.data.clone().cpu())
                if self.use_bias:
                    self.mean_est_record.append(self.netG.bias.data.clone().cpu())

            if (verbose < epochs) or (show):

                if ((ep+1) % verbose == 0):
                    if not self.use_bias:
                        print('Epoch:%d, LossD/G:%.4f/%.4f'
                                  % (ep+1, self.loss_D[-1], self.loss_G[-1]))
                    else:
                        print('Epoch:%d, LossD/G:%.4f/%.4f'
                                  % (ep+1, self.loss_D[-1], self.loss_G[-1]))

                    ## sample score
                    self.real_D = self.netD(self.Xtr.to(self.device))[1].detach().cpu().numpy()

                    ## generating sample score
                    if not self.elliptical:
                        if self.true_type == 'Student':
                            temp_z = torch.randn(10000, self.p).to(self.device) # [N, p]
                            temp_z.div_(torch.sqrt(self.t_chi2_d.sample((10000, 1))/self.t_df).to(self.device) + self.epsilon)
                            temp_g = self.netG(temp_z)
                        elif self.true_type == 'Gaussian':
                            temp_g = self.netG(torch.randn(10000, self.p).to(self.device))
                    
                    else:
                        temp_u = torch.randn(10000, self.p).to(self.device)
                        temp_u.data.div_(temp_u.norm(2, dim=1).view(-1, 1) + self.epsilon)
                        if inverse_gaussian:
                            temp_xi1 = torch.randn(10000, self.input_dim_G//2).to(self.device)
                            temp_xi2 = torch.randn(10000, self.input_dim_G//2).to(self.device)
                            temp_xi2.data = 1/(torch.abs(temp_xi2.data) + self.epsilon)
                            temp_xi = self.netGXi(torch.cat([temp_xi1, temp_xi2], dim=1)).view(10000, -1)
                        else:
                            temp_xi = self.netGXi(torch.randn(10000, self.input_dim_G).to(self.device)).view(10000, -1)
                        temp_g = self.netG(temp_u, temp_xi)
                    
                    self.gene_D = self.netD(temp_g)[1].detach().cpu().numpy()

                    sns.distplot(self.real_D[(self.real_D < 25) & (self.real_D > -25)], hist=False, label='Dist of Real Data')
                    sns.distplot(self.gene_D[(self.gene_D < 25) & (self.gene_D > -25)], hist=False, label='Dist of Generator')
                    plt.legend()
                    plt.title('Disc distribution')
                    plt.show()


        if self.use_bias:
            self.mean_avg = sum(self.mean_est_record)/len(self.mean_est_record)
        self.cov_avg = sum(self.cov_est_record)/len(self.cov_est_record)

        if show == True:

            plt.plot(self.loss_D)
            plt.title('loss_D')
            plt.show()
            
            plt.plot(self.loss_G)
            plt.title('loss_G')
            plt.show()

