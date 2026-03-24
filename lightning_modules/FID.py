from lightning import LightningModule
from lightning_modules.optimizer import make_optimizer, make_scheduler
from .loss import get_loss_funcs
from utils.debug_util import mkdir
from model import make_model
import torch
import numpy as np
import scipy.linalg
from abc import ABC


class Evaluator(ABC):
    def __init__(self):
        return

    def evaluate(self, dataloader):
        pass

class FIDModule(LightningModule, Evaluator):
    def __init__(self, model_cfg, train_cfg=None, loss_cfg=None, data_repr='gpos', name = 'FID'):
        super().__init__()
        self.name = name
        self.train_cfg = train_cfg
        self.save_hyperparameters()
        if loss_cfg is not None:
            self.loss_funcs = get_loss_funcs(loss_cfg)
            self.loss_weight = {key: loss_cfg[key]['weight'] for key in loss_cfg}
        else:
            self.loss_funcs = None

        if model_cfg.module != 'model.stgcn.STGCN':
            model_cfg.kwargs.input_dim = 66 if data_repr == 'gpos' else 135
        self.model = make_model(model_cfg)

        self.mean = None
        self.std = None
        self.features = []
        self.gt_mu = None
        self.gt_sigma = None
        self.true_positive = [0 for _ in range(12)]
        self.false_positive = [0 for _ in range(12)]
        self.cnt = [0 for _ in range(12)]

    def on_train_start(self) -> None:
        self.mean, self.std = self.trainer.train_dataloader.dataset.calc_stats()

    def configure_optimizers(self):
        optimizer = make_optimizer(self.train_cfg, self.model)
        scheduler = make_scheduler(self.train_cfg.scheduler, optimizer)
        return [optimizer], [scheduler]

    def _make_mask(self, bs, lengths, max_len):
        mask = torch.zeros([bs, max_len, 1], dtype=lengths.dtype, device=lengths.device)
        for i, l in enumerate(lengths):
            mask[i, :l] = 1
        return mask

    def training_step(self, batch, batch_idx):
        states, actions, lengths = batch
        actions -= 1
        actions = actions.squeeze(-1).to(torch.long)
        bs = states.shape[0]
        loss_dict = {}

        if self.hparams.model_cfg.module == 'model.extractor_model.TransformerEncoder':
            states = self.zscore_normalize(states)
            padding_mask = self._make_mask(bs, lengths, states.shape[1])
            padding_mask = (1. - padding_mask).to(bool)
            pred = self.model(states, padding_mask=padding_mask.squeeze(-1))
        elif self.hparams.model_cfg.module == 'model.extractor_model.MotionDiscriminator':
            states = self.zscore_normalize(states)
            pred = self.model(states, lengths)
        elif self.hparams.model_cfg.module == 'model.stgcn.STGCN':
            states = self.zscore_normalize(states)
            pred = self.model(states)

        if 'NLL Loss' in self.loss_funcs.keys():
            _, predicted = torch.max(torch.exp(pred), 1)
        else:
            _, predicted = torch.max(pred, 1)
        accuracy = (predicted == actions).sum().item() / actions.size(0)

        for key in self.loss_funcs:
            if 'Normalization' in key:
                loss_dict[key] = sum([param.pow(2).sum() for param in self.model.parameters()])
                continue
            loss_dict[key] = self.loss_funcs[key](pred, actions)
        loss = sum([loss_dict[key] * self.loss_weight[key] for key in loss_dict])
        self.log_dict(loss_dict, prog_bar=True)
        self.log('total_loss', loss, prog_bar=True)
        self.log('train_acc', accuracy, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        states, actions, lengths = batch
        actions -= 1
        actions = actions.squeeze(-1).to(torch.long)
        bs = states.shape[0]
        loss_dict = {}

        if self.hparams.model_cfg.module == 'model.extractor_model.TransformerEncoder':
            states = self.zscore_normalize(states)
            padding_mask = self._make_mask(bs, lengths, states.shape[1])
            padding_mask = (1. - padding_mask).to(bool)
            pred = self.model(states, padding_mask=padding_mask.squeeze(-1))
        elif self.hparams.model_cfg.module == 'model.extractor_model.MotionDiscriminator':
            states = self.zscore_normalize(states)
            pred = self.model(states, lengths)
        elif self.hparams.model_cfg.module == 'model.stgcn.STGCN':
            states = self.zscore_normalize(states)
            pred = self.model(states)

        for key in self.loss_funcs:
            if 'Normalization' in key:
                loss_dict[key] = sum([param.pow(2).sum() for param in self.model.parameters()])
                continue
            loss_dict[key] = self.loss_funcs[key](pred, actions)

        loss = sum([loss_dict[key] * self.loss_weight[key] for key in loss_dict])

        if self.loss_funcs is not None and 'NLL Loss' in self.loss_funcs.keys():
            _, predicted = torch.max(torch.exp(pred), 1)
        else:
            _, predicted = torch.max(pred, 1)
        accuracy = (predicted == actions).sum().item() / actions.size(0)
        self.log_dict({'val_loss': loss, 'val_acc': accuracy}, prog_bar=True)

    def zscore_normalize(self, data):
        mean = torch.tensor(self.mean, device=data.device, dtype=data.dtype)
        std = torch.tensor(self.std, device=data.device, dtype=data.dtype)
        return (data - mean) / std

    def on_predict_start(self) -> None:
        self.true_positive = [0 for _ in range(12)]
        self.false_positive = [0 for _ in range(12)]
        self.cnt = [0 for _ in range(12)]
        # mkdir('./exps/model_stats/FID')
        # np.save('./exps/model_stats/FID/Mean_test.npy', self.mean['predict'])
        # np.save('./exps/model_stats/FID/Std_test.npy', self.std['predict'])
        self.features = []
        self.gt_mu = None
        self.gt_sigma = None

    def predict_step(self, batch, batch_idx):
        states, actions, lengths = batch
        bs = states.shape[0]
        actions -= 1
        actions = actions.squeeze(-1).to(torch.long)
        padding_mask = self._make_mask(bs, lengths, states.shape[1])
        padding_mask = (1. - padding_mask).to(bool)
        states = self.zscore_normalize(states)
        if self.hparams.model_cfg.module == 'model.extractor_model.TransformerEncoder':
            padding_mask = self._make_mask(bs, lengths, states.shape[1])
            padding_mask = (1. - padding_mask).to(bool)
            feature = self.model.extract_features(states, padding_mask=padding_mask.squeeze(-1))
            pred = self.model(states, padding_mask=padding_mask.squeeze(- 1))
        elif self.hparams.model_cfg.module == 'model.extractor_model.MotionDiscriminator':
            feature = self.model.extract_features(states, lengths)
            pred = self.model(states, lengths)
        self.features.append(feature)

        _, predicted = torch.max(pred, 1)
        accuracy = (predicted == actions).sum().item() / actions.size(0)

        return accuracy


    def on_predict_end(self):
        self.features = torch.cat(self.features, dim=0)
        if self.gt_mu is None and self.gt_sigma is None:
            features = self.features.cpu().numpy()
            self.gt_mu = np.mean(features, axis=0)
            self.gt_sigma = np.cov(features, rowvar=False)

    @torch.no_grad()
    def evaluate(self, dataloader):
        assert self.gt_mu is not None and self.gt_sigma is not None
        # np.save('gt_mu.npy', self.gt_mu)
        # np.save('gt_sigma.npy', self.gt_sigma)
        self.model.eval()
        dataloader.dataset.metric_name = self.name
        self.features = []
        acc_mean = 0
        for idx, data in enumerate(dataloader):
            acc = self.predict_step(data, idx)
            acc_mean = acc_mean * (idx / (idx + 1)) + acc * (1 / (idx + 1))
        self.features = torch.cat(self.features, dim=0)
        features = self.features.cpu().numpy()
        features_clean = features[~np.isnan(features).any(axis=1)]
        diversity_times = 20
        diversity = 0
        num_motions = features_clean.shape[0]
        first_indices = np.random.randint(0, num_motions, diversity_times)
        second_indices = np.random.randint(0, num_motions, diversity_times)
        for first_idx, second_idx in zip(first_indices, second_indices):
            diversity += torch.dist(torch.tensor(features_clean[first_idx, :]),
                                    torch.tensor(features_clean[second_idx, :]))
        diversity /= diversity_times
        mu = np.mean(features_clean, axis=0)
        sigma = np.cov(features_clean, rowvar=False)

        fid = self.calc_fid(mu, self.gt_mu, sigma, self.gt_sigma)

        return {
            'FID': fid,
            'acc': acc_mean,
            'Diversity': diversity.item(),
        }

    @staticmethod
    def calc_fid(mu1, mu2, sigma1, sigma2, eps=1e-6):
        """
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
	    and X_2 ~ N(mu_2, C_2) is
			d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

