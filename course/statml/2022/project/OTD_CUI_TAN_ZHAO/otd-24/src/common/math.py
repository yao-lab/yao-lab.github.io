import torch


def kl_gaussian(mu, sigma, mu1=None, sigma1=None):
    """
    return kl ( N(mu, exp(0.5*sigma)) || N(mu1,sigma1) )
    if mu1, sigma1 == None
        use  (mu1, sigma1 ) = (0,I)
    """
    if mu1 is None:
        return 0.5 * torch.sum(-1 - 2 * torch.log(sigma) + mu ** 2 + sigma ** 2, dim=1)

    else:
        return 0.5 * torch.sum(
            -1
            - 2 * torch.log(sigma)
            + 2 * torch.log(sigma1)
            + ((mu - mu1) ** 2 + sigma ** 2) / sigma1 ** 2,
            dim=1)


def conditional_marginal_divergence_gaussian(f, s):
    # compute KL(q(z|x) \| q(z)) for gaussian q(z|x)

    dim_z = f.shape[1]
    batch_size = f.shape[0]

    all_pairs_GKL = all_pairs_gaussian_kl(f, s, dim_z)
    return all_pairs_GKL.mean(dim=1)


# 2*KL(N_0|N_1) = tr(\sigma_1^{-1} \sigma_0) +
#  (\mu_1 - \mu_0)\sigma_1^{-1}(\mu_1 - \mu_0) - k +
#  \log( \frac{\det \sigma_1}{\det \sigma_0} )
def all_pairs_gaussian_kl(mu, sigma, dim_z):
    # mu is [batchsize x dim_z]
    # sigma is [batchsize x dim_z]

    sigma_sq = sigma * sigma + 1e-8
    sigma_sq_inv = 1 / sigma_sq
    # sigma_inv is [batchsize x sizeof(latent_space)]

    # first term
    # dot product of all sigma_inv vectors with sigma
    # is the same as a matrix mult of diag
    first_term = torch.matmul(sigma_sq, sigma_sq_inv.transpose(1, 0))

    # second term (we break the mu_i-mu_j square term)
    # REMEMBER THAT THIS IS SIGMA_1, not SIGMA_0
    sqi = torch.matmul(mu * mu, sigma_sq_inv.transpose(1, 0))
    # sqi is now [batchsize x batchsize] = sum(mu[:,i]**2 / Sigma[j])

    sqj = mu * mu * sigma_sq_inv
    sqj = torch.sum(sqj, dim=1)
    # sqj is now [batchsize, 1] = mu[j]**2 / Sigma[j]

    # squared distance
    # (mu[i] - mu[j])\sigma_inv(mu[i] - mu[j]) = r[i] - 2*mu[i]*mu[j] + r[j]
    # uses broadcasting
    second_term = 2 * torch.matmul(mu, torch.transpose(mu * sigma_sq_inv, 1, 0))
    second_term = sqi + sqj.view(1, -1) - second_term

    # third term

    # log det A = tr log A
    # log \frac{ det \Sigma_1 }{ det \Sigma_0 } =
    #   \tr\log \Sigma_1 - \tr\log \Sigma_0
    # for each sample, we have B comparisons to B other samples...
    #   so this cancels out, but we keep it

    logi = 2 * torch.sum(torch.log(sigma), dim=1)
    logi = torch.reshape(logi, [-1, 1])
    logj = torch.reshape(logi, [1, -1])
    third_term = logi - logj

    # combine and return
    return 0.5 * (first_term + second_term + third_term - dim_z)
