import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.Encoder import Encoder


class InfoCTM(nn.Module):
    '''
        InfoCTM: A Mutual Information Maximization Perspective of Cross-lingual Topic Modeling. AAAI 2023

        Xiaobao Wu, Xinshuai Dong, Thong Nguyen, Chaoqun Liu, Liangming Pan, Anh Tuan Luu
    '''
    def __init__(self, trans_e2c, pretrain_word_embeddings_en, pretrain_word_embeddings_cn, vocab_size_en, vocab_size_cn, num_topics=50, en_units=200, dropout=0., temperature=0.2, pos_threshold=0.4, weight_MI=30.0):
        super().__init__()

        self.num_topics = num_topics

        self.encoder_en = Encoder(vocab_size_en, num_topics, en_units, dropout)
        self.encoder_cn = Encoder(vocab_size_cn, num_topics, en_units, dropout)

        self.a = 1 * np.ones((1, int(num_topics))).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T), requires_grad=False)
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T), requires_grad=False)

        self.decoder_bn_en = nn.BatchNorm1d(vocab_size_en, affine=True)
        self.decoder_bn_en.weight.requires_grad = False
        self.decoder_bn_cn = nn.BatchNorm1d(vocab_size_cn, affine=True)
        self.decoder_bn_cn.weight.requires_grad = False

        self.phi_en = nn.Parameter(nn.init.xavier_uniform_(torch.empty((num_topics, vocab_size_en))))
        self.phi_cn = nn.Parameter(nn.init.xavier_uniform_(torch.empty((num_topics, vocab_size_cn))))

        self.TAMI = TAMI(temperature, weight_MI, pos_threshold, trans_e2c, pretrain_word_embeddings_en, pretrain_word_embeddings_cn)

    def get_beta(self):
        beta_en = self.phi_en
        beta_cn = self.phi_cn
        return beta_en, beta_cn

    def get_theta(self, x, lang):
        theta, mu, logvar = getattr(self, f'encoder_{lang}')(x)

        if self.training:
            return theta, mu, logvar
        else:
            return mu

    def decode(self, theta, beta, lang):
        bn = getattr(self, f'decoder_bn_{lang}')
        d1 = F.softmax(bn(torch.matmul(theta, beta)), dim=1)
        return d1

    def forward(self, x_en, x_cn):
        theta_en, mu_en, logvar_en = self.get_theta(x_en, lang='en')
        theta_cn, mu_cn, logvar_cn = self.get_theta(x_cn, lang='cn')

        beta_en, beta_cn = self.get_beta()

        loss = 0.

        x_recon_en = self.decode(theta_en, beta_en, lang='en')
        x_recon_cn = self.decode(theta_cn, beta_cn, lang='cn')
        loss_en = self.compute_loss_TM(x_recon_en, x_en, mu_en, logvar_en)
        loss_cn = self.compute_loss_TM(x_recon_cn, x_cn, mu_cn, logvar_cn)

        loss = loss_en + loss_cn

        fea_en = beta_en.T
        fea_cn = beta_cn.T
        loss_TAMI = self.TAMI(fea_en, fea_cn)

        loss += loss_TAMI

        rst_dict = {
            'loss': loss,
            'loss_TAMI': loss_TAMI,
            'loss_en': loss_en,
            'loss_cn': loss_cn
        }

        return rst_dict

    def compute_loss_TM(self, recon_x, x, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.num_topics)

        RECON = -(x * (recon_x + 1e-10).log()).sum(1)

        LOSS = (RECON + KLD).mean()
        return LOSS

class TAMI(nn.Module):
    '''
        InfoCTM: A Mutual Information Maximization Perspective of Cross-lingual Topic Modeling. AAAI 2023

        Xiaobao Wu, Xinshuai Dong, Thong Nguyen, Chaoqun Liu, Liangming Pan, Anh Tuan Luu
    '''
    def __init__(self, temperature, weight_MI, pos_threshold, trans_e2c, pretrain_word_embeddings_en, pretrain_word_embeddings_cn):
        super().__init__()
        self.temperature = temperature
        self.weight_MI = weight_MI
        self.pos_threshold = pos_threshold
        self.pretrain_word_embeddings_en = pretrain_word_embeddings_en
        self.pretrain_word_embeddings_cn = pretrain_word_embeddings_cn

        self.trans_e2c = torch.as_tensor(trans_e2c).float()
        self.trans_e2c = nn.Parameter(self.trans_e2c, requires_grad=False)
        self.trans_c2e = self.trans_e2c.T

        pos_trans_mask_en, pos_trans_mask_cn, neg_trans_mask_en, neg_trans_mask_cn = self.compute_pos_neg(pretrain_word_embeddings_en, pretrain_word_embeddings_cn, self.trans_e2c, self.trans_c2e)
        self.pos_trans_mask_en = nn.Parameter(pos_trans_mask_en, requires_grad=False)
        self.pos_trans_mask_cn = nn.Parameter(pos_trans_mask_cn, requires_grad=False)
        self.neg_trans_mask_en = nn.Parameter(neg_trans_mask_en, requires_grad=False)
        self.neg_trans_mask_cn = nn.Parameter(neg_trans_mask_cn, requires_grad=False)

    def build_CVL_mask(self, embeddings):
        norm_embed = F.normalize(embeddings)
        cos_sim = torch.matmul(norm_embed, norm_embed.T)
        pos_mask = (cos_sim >= self.pos_threshold).float()
        return pos_mask

    def translation_mask(self, mask, trans_dict_matrix):
        # V1 x V2
        trans_mask = torch.matmul(mask, trans_dict_matrix)
        return trans_mask

    def compute_pos_neg(self, pretrain_word_embeddings_en, pretrain_word_embeddings_cn, trans_e2c, trans_c2e):
        # Ve x Ve
        pos_mono_mask_en = self.build_CVL_mask(torch.as_tensor(pretrain_word_embeddings_en))
        # Vc x Vc
        pos_mono_mask_cn = self.build_CVL_mask(torch.as_tensor(pretrain_word_embeddings_cn))

        # Ve x Vc
        pos_trans_mask_en = self.translation_mask(pos_mono_mask_en, trans_e2c)
        pos_trans_mask_cn = self.translation_mask(pos_mono_mask_cn, trans_c2e)

        neg_trans_mask_en = (pos_trans_mask_en <= 0).float()
        neg_trans_mask_cn = (pos_trans_mask_cn <= 0).float()

        return pos_trans_mask_en, pos_trans_mask_cn, neg_trans_mask_en, neg_trans_mask_cn

    def MutualInfo(self, anchor_feature, contrast_feature, mask, neg_mask):
        anchor_dot_contrast = torch.div(
            torch.matmul(F.normalize(anchor_feature, dim=1), F.normalize(contrast_feature, dim=1).T),
            self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * neg_mask
        sum_exp_logits = exp_logits.sum(1, keepdim=True)

        log_prob = logits - torch.log(sum_exp_logits + torch.exp(logits) + 1e-10)
        mean_log_prob = -(mask * log_prob).sum()
        return mean_log_prob

    def forward(self, fea_en, fea_cn):
        loss_TAMI = self.MutualInfo(fea_en, fea_cn, self.pos_trans_mask_en, self.neg_trans_mask_en)
        loss_TAMI += self.MutualInfo(fea_cn, fea_en, self.pos_trans_mask_cn, self.neg_trans_mask_cn)

        # Prevent division by zero
        total_pos_masks = self.pos_trans_mask_en.sum() + self.pos_trans_mask_cn.sum()
        loss_TAMI = loss_TAMI / torch.clamp(total_pos_masks, min=1.0)

        loss_TAMI = self.weight_MI * loss_TAMI
        return loss_TAMI