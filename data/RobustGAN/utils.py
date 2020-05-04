from scipy.stats import kendalltau, norm, t
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.io as sio

def kendall(Y, true, df=None, subsample=None):
    # X is torch.tensor N by p
    X = Y.cpu().numpy()
    # scaling factor
    medX = np.median(X, axis=0)
    X = X - medX
    # median absolute deviation
    s = np.median(np.abs(X), axis=0)
    # scatter = k * MAD with k = 1/F^{-1}(3/4), where F is dist of real
    if true == 'Gaussian':
        k = 1/norm.ppf(3/4)
    elif true == 'Student':
        assert df is not None
        k = 1/t.ppf(3/4, df=df)
    s = k * s
    # sub-sampling
    if subsample is not None:
        assert subsample <= len(X)
        indices = np.random.choice(len(X), size=subsample, replace=False)
        X = X[indices]
    _, p = X.shape
    corr = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1):
            corr[i, j] = np.sin(np.pi / 2 * kendalltau(Y[:, i], Y[:, j])[0])
            corr[j, i] = corr[i, j]
    cov = s.reshape(p, 1) * corr * s.reshape(1, p)
    return cov

# snpdict = sio.loadmat('/home/wzhuai/Robust/SNP500/dataset/snp452.mat')
# copinfo = snpdict['stock'][0]
# copname = [copinfo[i][0][0][1][0][1:-1] for i in range(len(copinfo))]
# copctg = [copinfo[i][0][0][2][0][1:-1] for i in range(len(copinfo))]
# prc = snpdict['X']
# logdiff = np.log(prc[1:]/prc[0:-1])

# # Top
# select_top = ['Microsoft', 'Apple', 'Amazon', 'Facebook', 'Berkshire',
#           'Alphabet', 'Johnson', 'JPMorgan', 'Exxon', 'Visa',
#           'Bank of America', 'Procter', 'Intel', 'Cisco', 'Verizon',
#           'AT&T', 'Home Depot', 'Chevron', 'Walt Disney', 'Pfizer',
#           'Mastercard', 'UnitedHealth', 'Boeing', 'Merck', 'Wells Fargo',
#           'Coca-Cola']

# select_lv = ['Microsoft', 'Apple', 'Amazon', 'JPMorgan',
#              'NIKE', 'Texas Ins', 'Costco',
#              'Charles', 'Southern Company', 'Deere',
#              'Equinix', 'Aflac', 'Valero',
#              'Halliburt', 'Ingersoll', 'Corning',
#              'Tyson', 'Realty', 'Edison',
#              'Keysight', 'Hess', 'Maxim Integrated',
#              'Hormel', 'Cboe Global', 'Alliant',
#              'Western Union', 'Interpublic', 'Mohawk',
#              'Discovery', 'Mattel', 'Macerich']
# selectcop_top = {}
# for name in select_top:
#     s = [name in copname[i] for i in range(len(copname))]
#     try:
#         idx = s.index(True)
#     except:
#         continue
#     selectcop_top[copname[idx]] = idx

# selectcop_lv = {}
# for name in select_lv:
#     s = [name in copname[i] for i in range(len(copname))]
#     try:
#         idx = s.index(True)
#     except:
#         continue
#     selectcop_lv[copname[idx]] = idx

# snp500 = {'logdiff':logdiff, 'copname':copname, 'copctg':copctg,
#           'topcop':selectcop_top, 'lvcop':selectcop_lv}
# with open('/home/wzhuai/Robust/SNP500/dataset/snp500.pkl', 'wb') as outF:
#     pickle.dump(snp500, outF)
