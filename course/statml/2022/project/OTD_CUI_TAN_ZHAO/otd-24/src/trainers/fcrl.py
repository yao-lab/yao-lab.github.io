import logging
import math
import wandb
import numpy as np
import torch
from box import Box
from sklearn.metrics import roc_auc_score

from lib import BaseTrainer
from src.common import metrics
from src.common.viz import do_classification

logger = logging.getLogger()

class FCRLTrainer(BaseTrainer):

    def compute_encoding(self, loader):
        # This function expects a trainer, and data loader and extracts representations for CVIB,
        # and other FCRL like models
        # return dict of np array

        n = loader.dataset.tensors[0].shape[0]

        mu = np.zeros((n, self.model.z_size))
        sigma = np.zeros((n, self.model.z_size))
        y = np.zeros(n)
        dim_c = math.ceil(math.log2(self.model.c_size)) if self.model.c_type == "binary" else self.model.c_mix_num

        c = np.zeros((n, dim_c))
        # NOTE: Only working for binary and one_hot encoding
        z = np.zeros((n, self.model.z_size))

        batch_size = loader.batch_size
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                x_ = batch[0].to(self.model.device)
                #modify here to get c_mix instead
                c_ = batch[-1].to(self.model.device)

                temp = self.model.forward(batch)
                f, s = temp["f"], temp["s"]
                mu[i * batch_size: (i + 1) * batch_size] = f.cpu().data.numpy()
                sigma[i * batch_size: (i + 1) * batch_size] = s.cpu().data.numpy()

                # batch has y labels
                if len(batch) > 2:
                    y[i * batch_size: (i + 1) * batch_size] = batch[2].view(-1).cpu().data.numpy()
                c[i * batch_size: (i + 1) * batch_size] = c_.cpu().data.numpy().reshape(-1, dim_c)
                z[i * batch_size: (i + 1) * batch_size] = self.model._sample(f,
                                                                             s).cpu().data.numpy()

        result = Box({"mu": mu, "sigma": sigma, "z": z, "c": c})

        if len(batch) > 2:
            result["y"] = y

        return result

    def validate(self, train_loader, valid_loader, *args, **kwargs):
        loss, aux_loss = super().validate(train_loader, valid_loader)
        return loss, aux_loss

    def test(self, train_loader, test_loader, *args, **kwargs):
        # here we will compute stats for both train and test data

        logger.info("Computing loss stats for train data")
        train_loss, train_aux_loss = super().validate(self, train_loader, train_loader)

        logger.info("Computing loss stats for test data")
        loss, aux_loss = super().validate(self, train_loader, test_loader)

        logger.info("Computing encoding for train data")
        temp = self.compute_encoding(train_loader)
        mu_train, sigma_train, z_train, c_train = temp.mu, temp.sigma, temp.z, temp.c
        y_train = temp.get("y")

        logger.info("Computing encoding for test data")
        temp = self.compute_encoding(test_loader)
        mu_test, sigma_test, z_test, c_test = temp.mu, temp.sigma, temp.z, temp.c
        y_test = temp.get("y")

        logger.info("Saving encoding")
        np.save(f"{self.result_dir}/embedding.npy", {"mu_train": mu_train,
                                                     "mu_test": mu_test,
                                                     "sigma_train": sigma_train,
                                                     "sigma_test": sigma_test,
                                                     "z_train": z_train,
                                                     # store it just in case we need to reproduce
                                                     "z_test": z_test,
                                                     # store it just in case we need to reproduce
                                                     "c_train": c_train,
                                                     "c_test": c_test,
                                                     "y_train": y_train,
                                                     "y_test": y_test,
                                                     })

        # train a classifier on representation and compute accuracy
        if y_train is not None and y_test is not None:
            # samples
            logger.info("Train classifier on representation")
            # random forest
            logger.info("Random forest classifier")
            acc,f1, rf_prob = do_classification(z_train, y_train, z_test, y_test, bb='rf')
            aux_loss.update({"rf_acc_sample": acc,"rf_f1_sample":f1})

            logger.info("AdaBoost classifier")
            acc,f1, ada_prob = do_classification(z_train, y_train, z_test, y_test, bb='ada')
            aux_loss.update({"ada_acc_sample": acc,"ada_f1_sample":f1})

            logger.info("Logistic regression classifier")
            acc,f1, lr_prob = do_classification(z_train, y_train, z_test, y_test, bb='lr')
            aux_loss.update({"lr_acc_sample": acc,"lr_f1_sample":f1})
            # neural network
            logger.info("MLP classifier")
            acc,f1, mlp_prob = do_classification(z_train, y_train, z_test, y_test, bb='mlp')
            aux_loss.update({"mlp_acc_sample": acc,"mlp_f1_sample":f1})

            if self.model.y_type == "binary":

                dp_rf_s=[]
                dp_rf_ss=[]
                dp_ada_s=[]
                dp_ada_ss=[]
                dp_lr_s=[]
                dp_lr_ss=[]
                dp_mlp_s=[]
                dp_mlp_ss=[]
                if type(self.model.c_size)!=int:
                    # compute c_size from c_mix
                    for i in range(self.model.c_mix_num):
                        #modify here to compute deltadp
                        dp_mlp_ss_tmp, _ = metrics.demographic_parity_difference_soft(y_test, c_test[:,i], mlp_prob)
                        dp_mlp_s_tmp, _ = metrics.demographic_parity_difference(y_test, c_test[:,i], mlp_prob)

                        dp_rf_ss_tmp, _ = metrics.demographic_parity_difference_soft(y_test, c_test[:,i], rf_prob)
                        dp_rf_s_tmp, _ = metrics.demographic_parity_difference(y_test, c_test[:,i], rf_prob)

                        dp_ada_ss_tmp, _ = metrics.demographic_parity_difference_soft(y_test, c_test[:,i], ada_prob)
                        dp_ada_s_tmp, _ = metrics.demographic_parity_difference(y_test, c_test[:,i], ada_prob)

                        dp_lr_ss_tmp, _ = metrics.demographic_parity_difference_soft(y_test, c_test[:,i], lr_prob)
                        dp_lr_s_tmp, _ = metrics.demographic_parity_difference(y_test, c_test[:,i], lr_prob)

                        dp_mlp_ss.append(dp_mlp_ss_tmp)
                        dp_mlp_s.append(dp_mlp_s_tmp)
                        dp_rf_ss.append(dp_rf_ss_tmp)
                        dp_rf_s.append(dp_rf_s_tmp)
                        dp_ada_ss.append(dp_ada_ss_tmp)
                        dp_ada_s.append(dp_ada_s_tmp)
                        dp_lr_ss.append(dp_lr_ss_tmp)
                        dp_lr_s.append(dp_lr_s_tmp)
                else:
                    dp_mlp_ss_tmp, _ = metrics.demographic_parity_difference_soft(y_test, c_test, mlp_prob)
                    dp_mlp_s_tmp, _ = metrics.demographic_parity_difference(y_test, c_test, mlp_prob)

                    dp_rf_ss_tmp, _ = metrics.demographic_parity_difference_soft(y_test, c_test, rf_prob)
                    dp_rf_s_tmp, _ = metrics.demographic_parity_difference(y_test, c_test, rf_prob)

                    dp_ada_ss_tmp, _ = metrics.demographic_parity_difference_soft(y_test, c_test, ada_prob)
                    dp_ada_s_tmp, _ = metrics.demographic_parity_difference(y_test, c_test, ada_prob)

                    dp_lr_ss_tmp, _ = metrics.demographic_parity_difference_soft(y_test, c_test, lr_prob)
                    dp_lr_s_tmp, _ = metrics.demographic_parity_difference(y_test, c_test, lr_prob)

                    dp_mlp_ss.append(dp_mlp_ss_tmp)
                    dp_mlp_s.append(dp_mlp_s_tmp)
                    dp_rf_ss.append(dp_rf_ss_tmp)
                    dp_rf_s.append(dp_rf_s_tmp)
                    dp_ada_ss.append(dp_ada_ss_tmp)
                    dp_ada_s.append(dp_ada_s_tmp)
                    dp_lr_ss.append(dp_lr_ss_tmp)
                    dp_lr_s.append(dp_lr_s_tmp)

                aux_loss.update({"demographic_parity_lr_sample": np.linalg.norm(dp_lr_s,ord=2)})
                aux_loss.update({"demographic_parity_lr_sample_soft": np.linalg.norm(dp_lr_ss,ord=2)})

                aux_loss.update({"demographic_parity_mlp_sample": np.linalg.norm(dp_mlp_s,ord=2)})
                aux_loss.update({"demographic_parity_mlp_sample_soft": np.linalg.norm(dp_mlp_ss,ord=2)})

                aux_loss.update({"demographic_parity_rf_sample": np.linalg.norm(dp_rf_s,ord=2)})
                aux_loss.update({"demographic_parity_rf_sample_soft": np.linalg.norm(dp_rf_ss,ord=2)})

                aux_loss.update({"demographic_parity_ada_sample": np.linalg.norm(dp_ada_s,ord=2)})
                aux_loss.update({"demographic_parity_ada_sample_soft": np.linalg.norm(dp_ada_ss,ord=2)})

                lr_auc_s = roc_auc_score(y_test, lr_prob[:, 1])
                mlp_auc_s = roc_auc_score(y_test, mlp_prob[:, 1])
                rf_auc_s = roc_auc_score(y_test, rf_prob[:, 1])
                ada_auc_s = roc_auc_score(y_test, ada_prob[:, 1])
                aux_loss.update({"lr_auc_sample": lr_auc_s, "mlp_auc_sample": mlp_auc_s,"rf_auc_sample": rf_auc_s, "ada_auc_sample": ada_auc_s})
            else:
                raise Exception("reminder: y_type is not binary and we need to check this")

        return loss, aux_loss
