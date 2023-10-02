import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import torch.nn.init as torch_init

from edl_loss import EvidenceLoss
from edl_loss import relu_evidence, exp_evidence, softplus_evidence

from einops import rearrange, repeat, reduce

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.kaiming_uniform_(m.weight)
        if type(m.bias) != type(None):
            m.bias.data.fill_(0)


class BWA_fusion_dropout_feat_v2(torch.nn.Module):
    def __init__(self, n_feature, n_class, **args):
        super().__init__()
        embed_dim = 1024
        self.bit_wise_attn = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, (3,), padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.channel_conv = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, (3,), padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 512, (3,), padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, (3,), padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Conv1d(512, 1, (1,)),
                                       nn.Dropout(0.5),
                                       nn.Sigmoid())
        self.channel_avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, vfeat, ffeat, is_training=False):
        channelfeat = self.channel_avg(vfeat)
        channel_attn = self.channel_conv(channelfeat)
        bit_wise_attn = self.bit_wise_attn(ffeat)
        filter_feat = torch.sigmoid(bit_wise_attn * channel_attn) * vfeat
        x_atn = self.attention(filter_feat)
        return x_atn, filter_feat

class DDG_Net(nn.Module):
    def __init__(self, n_feature, args):
        super().__init__()

        self.action_graph = nn.ModuleList(
            [nn.ModuleList([nn.Conv1d(n_feature, n_feature, (1,), padding=0) for _ in range(2)]) for _ in range(2)])

        self.background_graph = nn.ModuleList(
            [nn.ModuleList([nn.Conv1d(n_feature, n_feature, (1,), padding=0) for _ in range(2)]) for _ in range(2)])

        self.attentions = nn.ModuleList([nn.Sequential(nn.Conv1d(n_feature, 512, (3,), padding=1),
                                                       nn.LeakyReLU(0.2),
                                                       nn.Dropout(0.5),
                                                       nn.Conv1d(512, 512, (3,), padding=1),
                                                       nn.LeakyReLU(0.2),
                                                       nn.Conv1d(512, 1, (1,)),
                                                       nn.Dropout(0.5),
                                                       nn.Sigmoid()) for _ in range(2)])
        self.activation = nn.LeakyReLU(0.2)
        self.action_threshold = args['opt'].action_threshold
        self.background_threshold = args['opt'].background_threshold
        self.similarity_threshold = args['opt'].similarity_threshold
        self.temperature = args['opt'].temperature
        self.top_k_rat = args['opt'].top_k_rat
        self.weight = 1 / args['opt'].weight

    def forward(self, vfeat, ffeat, is_training=True, **args):
        ori_vatn = self.attentions[0](vfeat)
        ori_fatn = self.attentions[1](ffeat)  # B, 1, T

        action_mask, background_mask, ambiguous_mask, temp_mask, no_action_mask, no_background_mask \
            = self.action_background_mask(ori_vatn, ori_fatn)

        adjacency_action, adjacency_background, adjacency_ambiguous, adjacency_ambiguous_action, adjacency_ambiguous_background, adjacency_ambiguous_ambiguous, \
            = self.adjacency_matrix(vfeat, ffeat, action_mask, background_mask, ambiguous_mask,
                                    temp_mask)

        vfeat_avg = torch.matmul(vfeat, adjacency_action) + torch.matmul(vfeat, adjacency_background) + torch.matmul(
            vfeat, adjacency_ambiguous)

        action_vfeat_gcn = vfeat.clone()
        background_vfeat_gcn = vfeat.clone()
        for layer_a, layer_b in zip(self.action_graph[0], self.background_graph[0]):
            action_vfeat_gcn = self.activation(torch.matmul(layer_a(action_vfeat_gcn), adjacency_action))
            background_vfeat_gcn = self.activation(torch.matmul(layer_b(background_vfeat_gcn), adjacency_background))
        ambiguous_vfeat_gcn = torch.matmul(action_vfeat_gcn, adjacency_ambiguous_action) + \
                              torch.matmul(background_vfeat_gcn, adjacency_ambiguous_background) + \
                              torch.matmul(vfeat, adjacency_ambiguous_ambiguous)
        vfeat_gcn = action_vfeat_gcn + background_vfeat_gcn + ambiguous_vfeat_gcn

        new_vfeat = self.weight * vfeat + (1 - self.weight) * (vfeat_avg + vfeat_gcn) / 2

        ffeat_avg = torch.matmul(ffeat, adjacency_action) + torch.matmul(ffeat, adjacency_background) + torch.matmul(
            ffeat, adjacency_ambiguous)

        action_ffeat_gcn = ffeat.clone()
        background_ffeat_gcn = ffeat.clone()
        for layer_a, layer_b in zip(self.action_graph[1], self.background_graph[1]):
            action_ffeat_gcn = self.activation(torch.matmul(layer_a(action_ffeat_gcn), adjacency_action))
            background_ffeat_gcn = self.activation(torch.matmul(layer_b(background_ffeat_gcn), adjacency_background))
        ambiguous_ffeat_gcn = torch.matmul(action_ffeat_gcn, adjacency_ambiguous_action) + \
                              torch.matmul(background_ffeat_gcn, adjacency_ambiguous_background) + \
                              torch.matmul(ffeat, adjacency_ambiguous_ambiguous)
        ffeat_gcn = action_ffeat_gcn + background_ffeat_gcn + ambiguous_ffeat_gcn

        new_ffeat = self.weight * ffeat + (1 - self.weight) * (ffeat_avg + ffeat_gcn) / 2

        v_atn = self.attentions[0](new_vfeat)
        f_atn = self.attentions[1](new_ffeat)

        if is_training:
            loss = self.cp_loss(ori_vatn, ori_fatn, no_action_mask,
                                no_background_mask, ffeat_avg, ffeat_gcn, vfeat_avg, vfeat_gcn)

            return v_atn, new_vfeat, f_atn, new_ffeat, loss
        else:
            return v_atn, new_vfeat, f_atn, new_ffeat, {}

    def action_background_mask(self, f_atn, v_atn):

        T = f_atn.shape[2]

        action_row_mask = ((f_atn >= self.action_threshold) & (v_atn >= self.action_threshold)).to(torch.float)
        background_row_mask = ((f_atn < self.background_threshold) & (v_atn < self.background_threshold)).to(
            torch.float)

        action_background_row_mask = action_row_mask + background_row_mask
        ambiguous_row_mask = 1 - action_background_row_mask

        ambiguous_mask = torch.matmul(action_background_row_mask.transpose(-1, -2), ambiguous_row_mask)
        action_mask = torch.matmul(action_row_mask.transpose(-1, -2), action_row_mask)
        background_mask = torch.matmul(background_row_mask.transpose(-1, -2), background_row_mask)

        return action_mask, background_mask, ambiguous_mask, \
               action_background_row_mask.repeat(1, T, 1), action_row_mask == 0, background_row_mask == 0

    def adjacency_matrix(self, vfeat, ffeat, action_mask, background_mask, ambiguous_mask, temp_mask):
        """
        features"B,D,T
        """
        B = ffeat.shape[0]
        T = ffeat.shape[2]
        # graph
        f_feat = F.normalize(ffeat, dim=1)
        v_feat = F.normalize(vfeat, dim=1)
        v_similarity = torch.matmul(v_feat.transpose(1, 2), v_feat)
        f_similarity = torch.matmul(f_feat.transpose(1, 2), f_feat)

        fusion_similarity = (v_similarity + f_similarity) / 2

        # mask and normalize
        mask_value = 0
        fusion_similarity[fusion_similarity < self.similarity_threshold] = mask_value

        k = T // self.top_k_rat

        top_k_indices = torch.topk(fusion_similarity, T - k, dim=1, largest=False, sorted=False)[1]
        fusion_similarity = fusion_similarity.scatter(1, top_k_indices, mask_value)

        adjacency_action = fusion_similarity.masked_fill(action_mask == 0, mask_value)
        adjacency_background = fusion_similarity.masked_fill(background_mask == 0, mask_value)
        ambiguous_mask = (ambiguous_mask + torch.eye(T).masked_fill(temp_mask == 1, 0)) == 0
        adjacency_ambiguous = fusion_similarity.masked_fill(ambiguous_mask, mask_value)

        adjacency_action = F.normalize(adjacency_action, p=1, dim=1)
        adjacency_background = F.normalize(adjacency_background, p=1, dim=1)
        adjacency_ambiguous = F.normalize(adjacency_ambiguous, p=1, dim=1)

        ambiguous_action_mask = (action_mask.sum(dim=-1, keepdim=True) == 0).repeat(1, 1, T)
        adjacency_ambiguous_action = adjacency_ambiguous.masked_fill(ambiguous_action_mask, 0)

        ambiguous_background_mask = (background_mask.sum(dim=-1, keepdim=True) == 0).repeat(1, 1, T)
        adjacency_ambiguous_background = adjacency_ambiguous.masked_fill(ambiguous_background_mask, 0)

        adjacency_ambiguous_ambiguous = adjacency_ambiguous.masked_fill(torch.eye(T).unsqueeze(0).repeat(B, 1, 1) == 0,
                                                                        0)

        return adjacency_action, adjacency_background, adjacency_ambiguous, adjacency_ambiguous_action, adjacency_ambiguous_background, adjacency_ambiguous_ambiguous

    def cp_loss(self, ori_v_atn, ori_f_atn, no_action_mask,
                no_background_mask, ffeat_avg, ffeat_gcn, vfeat_avg, vfeat_gcn):

        action_mask = no_action_mask == False
        background_mask = no_background_mask == False

        ori_v_atn = ori_v_atn.detach()
        ori_f_atn = ori_f_atn.detach()
        ori_atn = (ori_f_atn + ori_v_atn) / 2

        # 4
        action_count = action_mask.sum(dim=-1).squeeze()
        background_count = background_mask.sum(dim=-1).squeeze()
        action_count = max(action_count.count_nonzero().item(), 1)
        background_count = max(background_count.count_nonzero().item(), 1)
        feat_loss = 0.5 * (
                (torch.sum(torch.exp(-(1 / ori_atn - 1) / self.temperature).masked_fill(no_action_mask,
                                                                                        0).detach() * torch.norm(
                    vfeat_avg - vfeat_gcn, dim=1, keepdim=True), dim=-1) /
                 torch.exp(-(1 / ori_atn - 1) / self.temperature).masked_fill(no_action_mask, 0).detach().sum(
                     dim=-1).clamp(min=1e-3)).sum() / action_count + \
                (torch.sum(torch.exp(-(1 / (1 - ori_atn) - 1) / self.temperature).masked_fill(no_background_mask,
                                                                                              0).detach() * torch.norm(
                    vfeat_avg - vfeat_gcn, dim=1, keepdim=True), dim=-1) /
                 torch.exp(-(1 / (1 - ori_atn) - 1) / self.temperature).masked_fill(no_background_mask, 0).detach().sum(
                     dim=-1).clamp(min=1e-3)).sum() / background_count) + \
                    0.5 * (
                            (torch.sum(torch.exp(-(1 / ori_atn - 1) / self.temperature).masked_fill(no_action_mask,
                                                                                                    0).detach() * torch.norm(
                                ffeat_avg - ffeat_gcn, dim=1, keepdim=True), dim=-1) /
                             torch.exp(-(1 / ori_atn - 1) / self.temperature).masked_fill(no_action_mask,
                                                                                          0).detach().sum(dim=-1).clamp(
                                 min=1e-3)).sum() / action_count + \
                            (torch.sum(
                                torch.exp(-(1 / (1 - ori_atn) - 1) / self.temperature).masked_fill(no_background_mask,
                                                                                                   0).detach() * torch.norm(
                                    ffeat_avg - ffeat_gcn, dim=1, keepdim=True), dim=-1) /
                             torch.exp(-(1 / (1 - ori_atn) - 1) / self.temperature).masked_fill(no_background_mask,
                                                                                                0).detach().sum(
                                 dim=-1).clamp(min=1e-3)).sum() / background_count)

        return {'feat_loss': feat_loss}


class DELU(torch.nn.Module):
    def __init__(self, n_feature, n_class, device="cuda:0", **args):
        super().__init__()
        embed_dim = 2048
        dropout_ratio = args['opt'].dropout_ratio
        self.device = device

        self.vAttn = getattr(models, args['opt'].AWM)(1024, args)
        self.fAttn = getattr(models, args['opt'].AWM)(1024, args)

        self.feat_encoder = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, (3,), padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_ratio)
        )

        self.fusion = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, (1,), padding=0),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_ratio)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(embed_dim, embed_dim, (3,), padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.7),
            nn.Conv1d(embed_dim, n_class + 1, (1,))
        )

        _kernel = 13
        self.pool = nn.AvgPool1d(_kernel, 1, padding=_kernel // 2, count_include_pad=True) \
            if _kernel is not None else nn.Identity()

        self.apply(weights_init)

    def forward(self, inputs, is_training=True, **args):
        feat = inputs.transpose(-1, -2)
        v_atn, vfeat = self.vAttn(feat[:, :1024, :], feat[:, 1024:, :])
        f_atn, ffeat = self.fAttn(feat[:, 1024:, :], feat[:, :1024, :])
        x_atn = (f_atn + v_atn) / 2
        nfeat = torch.cat((vfeat, ffeat), 1)
        nfeat = self.fusion(nfeat)
        x_cls = self.classifier(nfeat)

        if "ActivityNet" in args["opt"].dataset_name:
            x_cls = self.pool(x_cls)
            x_atn = self.pool(x_atn)
            f_atn = self.pool(f_atn)
            v_atn = self.pool(v_atn)

        outputs = {'feat': nfeat.transpose(-1, -2),
                   'cas': x_cls.transpose(-1, -2),
                   'attn': x_atn.transpose(-1, -2),
                   'v_atn': v_atn.transpose(-1, -2),
                   'f_atn': f_atn.transpose(-1, -2),
                   }

        return outputs

    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def criterion(self, outputs, labels, **args):
        feat, element_logits, element_atn = outputs['feat'], outputs['cas'], outputs['attn']
        v_atn = outputs['v_atn']
        f_atn = outputs['f_atn']
        mutual_loss = 0.5 * F.mse_loss(v_atn, f_atn.detach()) + 0.5 * F.mse_loss(f_atn, v_atn.detach())

        element_logits_supp = self._multiply(element_logits, element_atn, include_min=True)

        edl_loss = self.edl_loss(element_logits_supp,
                                 element_atn,
                                 labels,
                                 rat=args['opt'].rat_atn,
                                 n_class=args['opt'].num_class,
                                 epoch=args['itr'],
                                 total_epoch=args['opt'].max_iter,
                                 )

        uct_guide_loss = self.uct_guide_loss(element_logits,
                                             element_logits_supp,
                                             element_atn,
                                             v_atn,
                                             f_atn,
                                             n_class=args['opt'].num_class,
                                             epoch=args['itr'],
                                             total_epoch=args['opt'].max_iter,
                                             amplitude=args['opt'].amplitude,
                                             )

        loss_mil_orig, _ = self.topkloss(element_logits,
                                         labels,
                                         is_back=True,
                                         rat=args['opt'].k)

        # SAL
        loss_mil_supp, _ = self.topkloss(element_logits_supp,
                                         labels,
                                         is_back=False,
                                         rat=args['opt'].k)

        loss_3_supp_Contrastive = self.Contrastive(feat, element_logits_supp, labels, is_back=False)

        loss_norm = element_atn.mean()
        # guide loss
        loss_guide = (1 - element_atn -
                      element_logits.softmax(-1)[..., [-1]]).abs().mean()

        v_loss_norm = v_atn.mean()
        # guide loss
        v_loss_guide = (1 - v_atn -
                        element_logits.softmax(-1)[..., [-1]]).abs().mean()

        f_loss_norm = f_atn.mean()
        # guide loss
        f_loss_guide = (1 - f_atn -
                        element_logits.softmax(-1)[..., [-1]]).abs().mean()

        total_loss = (
                    args['opt'].alpha_edl * edl_loss +
                    args['opt'].alpha_uct_guide * uct_guide_loss +
                    loss_mil_orig.mean() + loss_mil_supp.mean() +
                    args['opt'].alpha3 * loss_3_supp_Contrastive +
                    args['opt'].alpha4 * mutual_loss +
                    args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3 +
                    args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3)

        loss_dict = {
            'edl_loss': args['opt'].alpha_edl * edl_loss,
            'uct_guide_loss': args['opt'].alpha_uct_guide * uct_guide_loss,
            'loss_mil_orig': loss_mil_orig.mean(),
            'loss_mil_supp': loss_mil_supp.mean(),
            'loss_supp_contrastive': args['opt'].alpha3 * loss_3_supp_Contrastive,
            'mutual_loss': args['opt'].alpha4 * mutual_loss,
            'norm_loss': args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3,
            'guide_loss': args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3,
            'total_loss': total_loss,
        }

        return total_loss, loss_dict

    def uct_guide_loss(self,
                       element_logits,
                       element_logits_supp,
                       element_atn,
                       v_atn,
                       f_atn,
                       n_class,
                       epoch,
                       total_epoch,
                       amplitude):

        evidence = exp_evidence(element_logits_supp)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=-1)
        snippet_uct = n_class / S

        total_snippet_num = element_logits.shape[1]
        curve = self.course_function(epoch, total_epoch, total_snippet_num, amplitude).to(self.device)

        loss_guide = (1 - element_atn - element_logits.softmax(-1)[..., [-1]]).abs().squeeze()

        v_loss_guide = (1 - v_atn - element_logits.softmax(-1)[..., [-1]]).abs().squeeze()

        f_loss_guide = (1 - f_atn - element_logits.softmax(-1)[..., [-1]]).abs().squeeze()

        total_loss_guide = (loss_guide + v_loss_guide + f_loss_guide) / 3

        _, uct_indices = torch.sort(snippet_uct, dim=1)
        sorted_curve = torch.gather(curve.repeat(10, 1), 1, uct_indices)

        uct_guide_loss = torch.mul(sorted_curve, total_loss_guide).mean()

        return uct_guide_loss

    def edl_loss(self,
                 element_logits_supp,
                 element_atn,
                 labels,
                 rat,
                 n_class,
                 epoch=0,
                 total_epoch=5000,
                 ):

        k = max(1, int(element_logits_supp.shape[-2] // rat))

        atn_values, atn_idx = torch.topk(
            element_atn,
            k=k,
            dim=1
        )
        atn_idx_expand = atn_idx.expand([-1, -1, n_class + 1])
        topk_element_logits = torch.gather(element_logits_supp, 1, atn_idx_expand)[:, :, :-1]
        video_logits = topk_element_logits.mean(dim=1)

        edl_loss = EvidenceLoss(
            num_classes=n_class,
            evidence='exp',
            loss_type='log',
            with_kldiv=False,
            with_avuloss=False,
            disentangle=False,
            annealing_method='exp')

        edl_results = edl_loss(
            output=video_logits,
            target=labels,
            epoch=epoch,
            total_epoch=total_epoch
        )

        edl_loss = edl_results['loss_cls'].mean()

        return edl_loss

    def course_function(self, epoch, total_epoch, total_snippet_num, amplitude):

        idx = torch.arange(total_snippet_num)
        theta = 2 * (idx + 0.5) / total_snippet_num - 1
        delta = - 2 * epoch / total_epoch + 1
        curve = amplitude * torch.tanh(theta * delta) + 1

        return curve

    def topkloss(self,
                 element_logits,
                 labels,
                 is_back=True,
                 rat=8):

        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)

        topk_val, topk_ind = torch.topk(
            element_logits,
            k=max(1, int(element_logits.shape[-2] // rat)),
            dim=-2)

        instance_logits = torch.mean(topk_val, dim=-2)

        labels_with_back = labels_with_back / (
                torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)

        milloss = - (labels_with_back * F.log_softmax(instance_logits, dim=-1)).sum(dim=-1)

        return milloss, topk_ind

    def Contrastive(self, x, element_logits, labels, is_back=False):
        if is_back:
            labels = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        sim_loss = 0.
        n_tmp = 0.
        _, n, c = element_logits.shape
        for i in range(0, 3 * 2, 2):
            atn1 = F.softmax(element_logits[i], dim=0)
            atn2 = F.softmax(element_logits[i + 1], dim=0)

            n1 = torch.FloatTensor([np.maximum(n - 1, 1)]).to(self.device)
            n2 = torch.FloatTensor([np.maximum(n - 1, 1)]).to(self.device)
            Hf1 = torch.mm(torch.transpose(x[i], 1, 0), atn1)  # (n_feature, n_class)
            Hf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), atn2)
            Lf1 = torch.mm(torch.transpose(x[i], 1, 0), (1 - atn1) / n1)
            Lf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), (1 - atn2) / n2)

            d1 = 1 - torch.sum(Hf1 * Hf2, dim=0) / (
                    torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))  # 1-similarity
            d2 = 1 - torch.sum(Hf1 * Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
            d3 = 1 - torch.sum(Hf2 * Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d2 + 0.5, torch.FloatTensor([0.]).to(self.device)) * labels[i, :] * labels[i + 1, :])
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d3 + 0.5, torch.FloatTensor([0.]).to(self.device)) * labels[i, :] * labels[i + 1, :])
            n_tmp = n_tmp + torch.sum(labels[i, :] * labels[i + 1, :])
        sim_loss = sim_loss / n_tmp
        return sim_loss

    def decompose(self, outputs, **args):
        feat, element_logits, atn_supp, atn_drop, element_atn = outputs

        return element_logits, element_atn


class DELU_ACT(torch.nn.Module):
    def __init__(self, n_feature, n_class, device="cuda:0", **args):
        super().__init__()
        embed_dim = 2048
        mid_dim = 1024
        self.device = device
        dropout_ratio = args['opt'].dropout_ratio
        reduce_ratio = args['opt'].reduce_ratio

        self.vAttn = getattr(models, args['opt'].AWM)(1024, args)
        self.fAttn = getattr(models, args['opt'].AWM)(1024, args)

        self.feat_encoder = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(dropout_ratio))
        self.fusion = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, 1, padding=0), nn.LeakyReLU(0.2), nn.Dropout(dropout_ratio))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Dropout(0.7), nn.Conv1d(embed_dim, n_class + 1, 1))
        # self.cadl = CADL()
        # self.attention = Non_Local_Block(embed_dim,mid_dim,dropout_ratio)

        self.channel_avg = nn.AdaptiveAvgPool1d(1)
        self.batch_avg = nn.AdaptiveAvgPool1d(1)
        self.ce_criterion = nn.BCELoss()
        _kernel = ((args['opt'].max_seqlen // args['opt'].t) // 2 * 2 + 1)
        self.pool = nn.AvgPool1d(_kernel, 1, padding=_kernel // 2, count_include_pad=True) \
            if _kernel is not None else nn.Identity()
        self.apply(weights_init)

    def forward(self, inputs, is_training=True, **args):
        feat = inputs.transpose(-1, -2)
        b, c, n = feat.size()
        # feat = self.feat_encoder(x)
        v_atn, vfeat = self.vAttn(feat[:, :1024, :], feat[:, 1024:, :])
        f_atn, ffeat = self.fAttn(feat[:, 1024:, :], feat[:, :1024, :])
        x_atn = (f_atn + v_atn) / 2
        nfeat = torch.cat((vfeat, ffeat), 1)
        nfeat = self.fusion(nfeat)
        x_cls = self.classifier(nfeat)
        if not is_training: # 只有测试才搞这个
            x_cls = self.pool(x_cls)
            x_atn = self.pool(x_atn)
            f_atn = self.pool(f_atn)
            v_atn = self.pool(v_atn)
        # fg_mask, bg_mask,dropped_fg_mask = self.cadl(x_cls, x_atn, include_min=True)

        return {'feat': nfeat.transpose(-1, -2), 'cas': x_cls.transpose(-1, -2), 'attn': x_atn.transpose(-1, -2),
                'v_atn': v_atn.transpose(-1, -2), 'f_atn': f_atn.transpose(-1, -2)}
        # ,fg_mask.transpose(-1, -2), bg_mask.transpose(-1, -2),dropped_fg_mask.transpose(-1, -2)
        # return att_sigmoid,att_logit, feat_emb, bag_logit, instance_logit

    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def criterion(self, outputs, labels, **args):
        feat, element_logits, element_atn = outputs['feat'], outputs['cas'], outputs['attn']
        v_atn = outputs['v_atn']
        f_atn = outputs['f_atn']
        mutual_loss = 0.5 * F.mse_loss(v_atn, f_atn.detach()) + 0.5 * F.mse_loss(f_atn, v_atn.detach())
        # learning weight dynamic, lambda1 (1-lambda1)
        b, n, c = element_logits.shape
        element_logits_supp = self._multiply(element_logits, element_atn, include_min=True)
        loss_mil_orig, _ = self.topkloss(element_logits,
                                         labels,
                                         is_back=True,
                                         rat=args['opt'].k,
                                         reduce=None)
        # SAL
        loss_mil_supp, _ = self.topkloss(element_logits_supp,
                                         labels,
                                         is_back=False,
                                         rat=args['opt'].k,
                                         reduce=None)

        edl_loss = self.edl_loss(element_logits_supp,
                                 element_atn,
                                 labels,
                                 rat=args['opt'].rat_atn,
                                 n_class=args['opt'].num_class,
                                 epoch=args['itr'],
                                 total_epoch=args['opt'].max_iter,
                                 )

        uct_guide_loss = self.uct_guide_loss(element_logits,
                                             element_logits_supp,
                                             element_atn,
                                             v_atn,
                                             f_atn,
                                             n_class=args['opt'].num_class,
                                             epoch=args['itr'],
                                             total_epoch=args['opt'].max_iter,
                                             amplitude=args['opt'].amplitude,
                                             )

        loss_3_supp_Contrastive = self.Contrastive(feat, element_logits_supp, labels, is_back=False)

        loss_norm = element_atn.mean()
        # guide loss
        loss_guide = (1 - element_atn -
                      element_logits.softmax(-1)[..., [-1]]).abs().mean()

        v_loss_norm = v_atn.mean()
        # guide loss
        v_loss_guide = (1 - v_atn -
                        element_logits.softmax(-1)[..., [-1]]).abs().mean()

        f_loss_norm = f_atn.mean()
        # guide loss
        f_loss_guide = (1 - f_atn -
                        element_logits.softmax(-1)[..., [-1]]).abs().mean()

        # total loss
        total_loss = (
                    args['opt'].alpha_edl * edl_loss +
                    args['opt'].alpha_uct_guide * uct_guide_loss +
                    loss_mil_orig.mean() + loss_mil_supp.mean() +
                    args['opt'].alpha3 * loss_3_supp_Contrastive +
                    args['opt'].alpha4 * mutual_loss +
                    args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3 +
                    args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3)

        loss_dict = {
            'edl_loss': args['opt'].alpha_edl * edl_loss,
            'uct_guide_loss': args['opt'].alpha_uct_guide * uct_guide_loss,
            'loss_mil_orig': loss_mil_orig.mean(),
            'loss_mil_supp': loss_mil_supp.mean(),
            'loss_supp_contrastive': args['opt'].alpha3 * loss_3_supp_Contrastive,
            'mutual_loss': args['opt'].alpha4 * mutual_loss,
            'norm_loss': args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3,
            'guide_loss': args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3,
            'total_loss': total_loss,
        }

        return total_loss, loss_dict

    def uct_guide_loss(self,
                       element_logits,
                       element_logits_supp,
                       element_atn,
                       v_atn,
                       f_atn,
                       n_class,
                       epoch,
                       total_epoch,
                       amplitude):

        evidence = exp_evidence(element_logits_supp)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=-1)
        snippet_uct = n_class / S

        total_snippet_num = element_logits.shape[1]
        curve = self.course_function(epoch, total_epoch, total_snippet_num, amplitude).to(self.device)

        loss_guide = (1 - element_atn - element_logits.softmax(-1)[..., [-1]]).abs().squeeze()

        v_loss_guide = (1 - v_atn - element_logits.softmax(-1)[..., [-1]]).abs().squeeze()

        f_loss_guide = (1 - f_atn - element_logits.softmax(-1)[..., [-1]]).abs().squeeze()

        total_loss_guide = (loss_guide + v_loss_guide + f_loss_guide) / 3

        _, uct_indices = torch.sort(snippet_uct, dim=1)
        sorted_curve = torch.gather(curve.repeat(10, 1), 1, uct_indices)

        uct_guide_loss = torch.mul(sorted_curve, total_loss_guide).mean()

        return uct_guide_loss

    def edl_loss(self,
                 element_logits_supp,
                 element_atn,
                 labels,
                 rat,
                 n_class,
                 epoch=0,
                 total_epoch=5000,
                 ):

        k = max(1, int(element_logits_supp.shape[-2] // rat))

        atn_values, atn_idx = torch.topk(
            element_atn,
            k=k,
            dim=1
        )

        atn_idx_expand = atn_idx.expand([-1, -1, n_class + 1])
        topk_element_logits = torch.gather(element_logits_supp, 1, atn_idx_expand)[:, :, :-1]

        video_logits = topk_element_logits.mean(dim=1)

        edl_loss = EvidenceLoss(
            num_classes=n_class,
            evidence='relu',
            loss_type='mse',
            with_kldiv=False,
            with_avuloss=False,
            disentangle=False,
            annealing_method='exp')

        edl_results = edl_loss(
            output=video_logits,
            target=labels,
            epoch=epoch,
            total_epoch=total_epoch
        )

        edl_loss = edl_results['loss_cls'].mean()

        return edl_loss

    def course_function(self, epoch, total_epoch, total_snippet_num, amplitude):

        idx = torch.arange(total_snippet_num)

        # From -1 to 1
        theta = 2 * (idx + 0.5) / total_snippet_num - 1

        # From 1 to -1
        delta = - 2 * epoch / total_epoch + 1

        curve = amplitude * torch.tanh(theta * delta) + 1

        return curve

    def topkloss(self,
                 element_logits,
                 labels,
                 is_back=True,
                 lab_rand=None,
                 rat=8,
                 reduce=None):

        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        if lab_rand is not None:
            labels_with_back = torch.cat((labels, lab_rand), dim=-1)

        topk_val, topk_ind = torch.topk(
            element_logits,
            k=max(1, int(element_logits.shape[-2] // rat)),
            dim=-2)
        instance_logits = torch.mean(
            topk_val,
            dim=-2,
        )
        labels_with_back = labels_with_back / (
                torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)
        milloss = (-(labels_with_back *
                     F.log_softmax(instance_logits, dim=-1)).sum(dim=-1))
        if reduce is not None:
            milloss = milloss.mean()
        return milloss, topk_ind

    def Contrastive(self, x, element_logits, labels, is_back=False):
        if is_back:
            labels = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        sim_loss = 0.
        n_tmp = 0.
        _, n, c = element_logits.shape
        for i in range(0, 3 * 2, 2):
            atn1 = F.softmax(element_logits[i], dim=0)
            atn2 = F.softmax(element_logits[i + 1], dim=0)

            n1 = torch.FloatTensor([np.maximum(n - 1, 1)]).to(self.device)
            n2 = torch.FloatTensor([np.maximum(n - 1, 1)]).to(self.device)
            Hf1 = torch.mm(torch.transpose(x[i], 1, 0), atn1)  # (n_feature, n_class)
            Hf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), atn2)
            Lf1 = torch.mm(torch.transpose(x[i], 1, 0), (1 - atn1) / n1)
            Lf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), (1 - atn2) / n2)

            d1 = 1 - torch.sum(Hf1 * Hf2, dim=0) / (
                    torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))  # 1-similarity
            d2 = 1 - torch.sum(Hf1 * Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
            d3 = 1 - torch.sum(Hf2 * Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d2 + 0.5, torch.FloatTensor([0.]).to(self.device)) * labels[i, :] * labels[i + 1, :])
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d3 + 0.5, torch.FloatTensor([0.]).to(self.device)) * labels[i, :] * labels[i + 1, :])
            n_tmp = n_tmp + torch.sum(labels[i, :] * labels[i + 1, :])
        sim_loss = sim_loss / n_tmp
        return sim_loss

    def decompose(self, outputs, **args):
        feat, element_logits, atn_supp, atn_drop, element_atn = outputs

        return element_logits, element_atn

class DELU_MULTI_SCALE(torch.nn.Module):
    def __init__(self, n_feature, n_class, device="cuda:0", **args):
        super().__init__()
        embed_dim = 2048
        mid_dim = 1024
        self.device = device
        dropout_ratio = args['opt'].dropout_ratio
        reduce_ratio = args['opt'].reduce_ratio

        self.vAttn = getattr(models, args['opt'].AWM)(1024, args)
        self.fAttn = getattr(models, args['opt'].AWM)(1024, args)

        self.scales = args['opt'].scales

        self.feat_encoder = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(dropout_ratio))
        self.fusion = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, 1, padding=0), nn.LeakyReLU(0.2), nn.Dropout(dropout_ratio))

        self.channel_avg = nn.AdaptiveAvgPool1d(1)
        self.batch_avg = nn.AdaptiveAvgPool1d(1)
        self.ce_criterion = nn.BCELoss()

        self.pool = nn.ModuleList()
        for _kernel in self.scales:
            # self.pool.append(nn.AvgPool1d(_kernel, 1, padding=_kernel // 2, count_include_pad=True))
            # self.fusions.append(nn.Conv1d(embed_dim, embed_dim, 1))
            self.pool.append(DirtyAvgPool(_kernel))

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Dropout(0.7), 
            nn.Conv1d(embed_dim, n_class + 1, 1))

        self.apply(weights_init)

    def pool_forward(self, x):

        head_output=[]
        current = self.pool[-1](x)
        pre = current
        head_output.append(current.transpose(-1, -2))

        for i in range(len(self.scales)-2,-1,-1):
            current = self.pool[i](x) # 这里后期再考虑各层配合
            head_output.append(current.transpose(-1, -2))

        return list(reversed(head_output)) # 1,2,4,8...

    def forward(self, inputs, is_training=True, **args):
        feat = inputs.transpose(-1, -2)
        b, c, n = feat.size()
        # feat = self.feat_encoder(x)
        v_atn, vfeat = self.vAttn(feat[:, :1024, :], feat[:, 1024:, :])
        f_atn, ffeat = self.fAttn(feat[:, 1024:, :], feat[:, :1024, :])
        x_atn = (f_atn + v_atn) / 2
        nfeat = torch.cat((vfeat, ffeat), 1)
        nfeat = self.fusion(nfeat)
        x_cls = self.classifier(nfeat)

        x_cls = self.pool_forward(x_cls)
        x_atn = self.pool_forward(x_atn)
        f_atn = self.pool_forward(f_atn)
        v_atn = self.pool_forward(v_atn)

        # fg_mask, bg_mask,dropped_fg_mask = self.cadl(x_cls, x_atn, include_min=True)

        return {'feat': nfeat.transpose(-1, -2), 'cas': x_cls, 'attn': x_atn,
                'v_atn': v_atn, 'f_atn': f_atn}

    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def criterion(self, outputs, labels, **args):
        total_loss = 0.0  # Initialize the total loss
        avg_loss_dict = {}  # Initialize the dictionary for averaging loss_dict elements

        for scale in range(len(self.scales)):
            single_scale_total_loss, single_scale_loss_dict = self.single_scale_criterion(scale, outputs, labels, args)
            
            # Accumulate the single_scale_total_loss
            total_loss += single_scale_total_loss

            # Aggregate elements from single_scale_loss_dict for averaging
            for key, value in single_scale_loss_dict.items():
                if key in avg_loss_dict:
                    avg_loss_dict[key] += value
                else:
                    avg_loss_dict[key] = value

        # Calculate the average loss
        num_scales = float(len(self.scales))
        avg_total_loss = total_loss / num_scales

        # Average the elements in avg_loss_dict
        for key in avg_loss_dict:
            avg_loss_dict[key] /= num_scales

        return avg_total_loss, avg_loss_dict
        

    def single_scale_criterion(self, scale, outputs, labels, args):
        feat, element_logits, element_atn = outputs['feat'], outputs['cas'][scale], outputs['attn'][scale]
        v_atn = outputs['v_atn'][scale]
        f_atn = outputs['f_atn'][scale]
        
        mutual_loss = 0.5 * F.mse_loss(v_atn, f_atn.detach()) + 0.5 * F.mse_loss(f_atn, v_atn.detach())
        # learning weight dynamic, lambda1 (1-lambda1)
        b, n, c = element_logits.shape
        element_logits_supp = self._multiply(element_logits, element_atn, include_min=True)
        loss_mil_orig, _ = self.topkloss(element_logits,
                                         labels,
                                         is_back=True,
                                         rat=max(args['opt'].k//max((self.scales[scale]-1),1), 1),
                                         reduce=None)
        # SAL
        loss_mil_supp, _ = self.topkloss(element_logits_supp,
                                         labels,
                                         is_back=False,
                                         rat=max(args['opt'].k//self.scales[scale], 1),
                                         reduce=None)
        

        edl_loss = self.edl_loss(element_logits_supp,
                                element_atn,
                                labels,
                                rat=args['opt'].rat_atn,
                                n_class=args['opt'].num_class,
                                epoch=args['itr'],
                                total_epoch=args['opt'].max_iter,
                                )

        uct_guide_loss = self.uct_guide_loss(element_logits,
                                            element_logits_supp,
                                            element_atn,
                                            v_atn,
                                            f_atn,
                                            n_class=args['opt'].num_class,
                                            epoch=args['itr'],
                                            total_epoch=args['opt'].max_iter,
                                            amplitude=args['opt'].amplitude,
                                            )

        loss_3_supp_Contrastive = self.Contrastive(feat, element_logits_supp, labels, is_back=False)

        loss_norm = element_atn.mean()
        # guide loss
        loss_guide = (1 - element_atn -
                    element_logits.softmax(-1)[..., [-1]]).abs().mean()

        v_loss_norm = v_atn.mean()
        # guide loss
        v_loss_guide = (1 - v_atn -
                        element_logits.softmax(-1)[..., [-1]]).abs().mean()

        f_loss_norm = f_atn.mean()
        # guide loss
        f_loss_guide = (1 - f_atn -
                        element_logits.softmax(-1)[..., [-1]]).abs().mean()

        # total loss
        total_loss = (
                    args['opt'].alpha_edl * edl_loss +
                    args['opt'].alpha_uct_guide * uct_guide_loss +
                    loss_mil_orig.mean() + loss_mil_supp.mean() +
                    args['opt'].alpha3 * loss_3_supp_Contrastive +
                    args['opt'].alpha4 * mutual_loss +
                    args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3 +
                    args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3)

        loss_dict = {
            'edl_loss': args['opt'].alpha_edl * edl_loss,
            'uct_guide_loss': args['opt'].alpha_uct_guide * uct_guide_loss,
            'loss_mil_orig': loss_mil_orig.mean(),
            'loss_mil_supp': loss_mil_supp.mean(),
            'loss_supp_contrastive': args['opt'].alpha3 * loss_3_supp_Contrastive,
            'mutual_loss': args['opt'].alpha4 * mutual_loss,
            'norm_loss': args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3,
            'guide_loss': args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3,
            'total_loss': total_loss,
        }

        return total_loss, loss_dict

    def uct_guide_loss(self,
                       element_logits,
                       element_logits_supp,
                       element_atn,
                       v_atn,
                       f_atn,
                       n_class,
                       epoch,
                       total_epoch,
                       amplitude):

        evidence = exp_evidence(element_logits_supp)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=-1)
        snippet_uct = n_class / S

        total_snippet_num = element_logits.shape[1]
        curve = self.course_function(epoch, total_epoch, total_snippet_num, amplitude).to(self.device)

        loss_guide = (1 - element_atn - element_logits.softmax(-1)[..., [-1]]).abs().squeeze()

        v_loss_guide = (1 - v_atn - element_logits.softmax(-1)[..., [-1]]).abs().squeeze()

        f_loss_guide = (1 - f_atn - element_logits.softmax(-1)[..., [-1]]).abs().squeeze()

        total_loss_guide = (loss_guide + v_loss_guide + f_loss_guide) / 3

        _, uct_indices = torch.sort(snippet_uct, dim=1)
        sorted_curve = torch.gather(curve.repeat(10, 1), 1, uct_indices)

        uct_guide_loss = torch.mul(sorted_curve, total_loss_guide).mean()

        return uct_guide_loss

    def edl_loss(self,
                 element_logits_supp,
                 element_atn,
                 labels,
                 rat,
                 n_class,
                 epoch=0,
                 total_epoch=5000,
                 ):

        k = max(1, int(element_logits_supp.shape[-2] // rat))

        atn_values, atn_idx = torch.topk(
            element_atn,
            k=k,
            dim=1
        )

        atn_idx_expand = atn_idx.expand([-1, -1, n_class + 1])
        topk_element_logits = torch.gather(element_logits_supp, 1, atn_idx_expand)[:, :, :-1]

        video_logits = topk_element_logits.mean(dim=1)

        edl_loss = EvidenceLoss(
            num_classes=n_class,
            evidence='relu',
            loss_type='mse',
            with_kldiv=False,
            with_avuloss=False,
            disentangle=False,
            annealing_method='exp')

        edl_results = edl_loss(
            output=video_logits,
            target=labels,
            epoch=epoch,
            total_epoch=total_epoch
        )

        edl_loss = edl_results['loss_cls'].mean()

        return edl_loss

    def course_function(self, epoch, total_epoch, total_snippet_num, amplitude):

        idx = torch.arange(total_snippet_num)

        # From -1 to 1
        theta = 2 * (idx + 0.5) / total_snippet_num - 1

        # From 1 to -1
        delta = - 2 * epoch / total_epoch + 1

        curve = amplitude * torch.tanh(theta * delta) + 1

        return curve

    def topkloss(self,
                 element_logits,
                 labels,
                 is_back=True,
                 lab_rand=None,
                 rat=8,
                 reduce=None):

        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        if lab_rand is not None:
            labels_with_back = torch.cat((labels, lab_rand), dim=-1)

        topk_val, topk_ind = torch.topk(
            element_logits,
            k=max(1, int(element_logits.shape[-2] // rat)),
            dim=-2)
        instance_logits = torch.mean(
            topk_val,
            dim=-2,
        )
        labels_with_back = labels_with_back / (
                torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)
        milloss = (-(labels_with_back *
                     F.log_softmax(instance_logits, dim=-1)).sum(dim=-1))
        if reduce is not None:
            milloss = milloss.mean()
        return milloss, topk_ind

    def Contrastive(self, x, element_logits, labels, is_back=False):
        if is_back:
            labels = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        sim_loss = 0.
        n_tmp = 0.
        _, n, c = element_logits.shape
        for i in range(0, 3 * 2, 2):
            atn1 = F.softmax(element_logits[i], dim=0)
            atn2 = F.softmax(element_logits[i + 1], dim=0)

            n1 = torch.FloatTensor([np.maximum(n - 1, 1)]).to(self.device)
            n2 = torch.FloatTensor([np.maximum(n - 1, 1)]).to(self.device)
            Hf1 = torch.mm(torch.transpose(x[i], 1, 0), atn1)  # (n_feature, n_class)
            Hf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), atn2)
            Lf1 = torch.mm(torch.transpose(x[i], 1, 0), (1 - atn1) / n1)
            Lf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), (1 - atn2) / n2)

            d1 = 1 - torch.sum(Hf1 * Hf2, dim=0) / (
                    torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))  # 1-similarity
            d2 = 1 - torch.sum(Hf1 * Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
            d3 = 1 - torch.sum(Hf2 * Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d2 + 0.5, torch.FloatTensor([0.]).to(self.device)) * labels[i, :] * labels[i + 1, :])
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d3 + 0.5, torch.FloatTensor([0.]).to(self.device)) * labels[i, :] * labels[i + 1, :])
            n_tmp = n_tmp + torch.sum(labels[i, :] * labels[i + 1, :])
        sim_loss = sim_loss / n_tmp
        return sim_loss

    def decompose(self, outputs, **args):
        feat, element_logits, atn_supp, atn_drop, element_atn = outputs

        return element_logits, element_atn

class DELU_PYRAMID(torch.nn.Module):
    def __init__(self, n_feature, n_class, device="cuda:0", **args):
        super().__init__()
        embed_dim = 2048
        dropout_ratio = args['opt'].dropout_ratio
        self.device = device

        self.vAttn = getattr(models, args['opt'].AWM)(1024, args)
        self.fAttn = getattr(models, args['opt'].AWM)(1024, args)

        self.scales = args['opt'].scales

        self.feat_encoder = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(dropout_ratio))
        self.fusion = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, 1, padding=0), nn.LeakyReLU(0.2), nn.Dropout(dropout_ratio))

        self.channel_avg = nn.AdaptiveAvgPool1d(1)
        self.batch_avg = nn.AdaptiveAvgPool1d(1)
        self.ce_criterion = nn.BCELoss()

        self.attentions = nn.ModuleList()
        self.attn_fusion = nn.ModuleList()
        self.pools = nn.ModuleList()

        for _kernel in self.scales: # 每个scale添加一个attention windows
            self.attentions.append(nn.AvgPool1d(hidden_dim=embed_dim, kernel=_kernel, dropout_prob=dropout_ratio))
            self.pools
            self.attn_fusion.append(nn.Sequential(
            nn.Conv1d(n_feature, n_feature, 1, padding=0), nn.LeakyReLU(0.2), nn.Dropout(dropout_ratio)))

        # self.pool = nn.AvgPool1d(13, 1, padding=13 // 2, count_include_pad=True) # 只保留单一的scale用于Anet

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Dropout(0.7), 
            nn.Conv1d(embed_dim, n_class + 1, 1))

        self.apply(weights_init)
    
    def attention_pool(self, x):
        head_output = []
        previous = self.attentions[-1](x) # the last scale
        current = None

        for i in range(len(self.scales) - 2, -1, -1):
            scale = self.scales[i]
            current = self.attentions[i](x)
            current = current + self.attn_fusion[i+1](previous) # 把上层和这层融合
            previous = current
        
        output = self.attn_fusion[0](previous) + x

        return output  # 1,2,4,8...

    def forward(self, inputs, is_training=True, **args):
        feat = inputs.transpose(-1, -2)
        b, c, n = feat.size()
        # feat = self.feat_encoder(x)
        v_atn, vfeat = self.vAttn(feat[:, :1024, :], feat[:, 1024:, :])
        f_atn, ffeat = self.fAttn(feat[:, 1024:, :], feat[:, :1024, :])
        x_atn = (f_atn + v_atn) / 2
        nfeat = torch.cat((vfeat, ffeat), 1)
        nfeat = self.fusion(nfeat)

        nfeat = self.attention_pool(nfeat) # 和普通的DELU唯一的区别

        x_cls = self.classifier(nfeat)

        '''
        if "ActivityNet" in args["opt"].dataset_name:
            x_cls = self.pool(x_cls)
            x_atn = self.pool(x_atn)
            f_atn = self.pool(f_atn)
            v_atn = self.pool(v_atn)
        '''

        # fg_mask, bg_mask,dropped_fg_mask = self.cadl(x_cls, x_atn, include_min=True)

        return {'feat': nfeat.transpose(-1, -2), 'cas': x_cls.transpose(-1, -2), 'attn': x_atn.transpose(-1, -2),
                'v_atn': v_atn.transpose(-1, -2), 'f_atn': f_atn.transpose(-1, -2)}

    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def criterion(self, outputs, labels, **args):
        feat, element_logits, element_atn = outputs['feat'], outputs['cas'], outputs['attn']
        v_atn = outputs['v_atn']
        f_atn = outputs['f_atn']
        mutual_loss = 0.5 * F.mse_loss(v_atn, f_atn.detach()) + 0.5 * F.mse_loss(f_atn, v_atn.detach())
        # learning weight dynamic, lambda1 (1-lambda1)
        b, n, c = element_logits.shape
        element_logits_supp = self._multiply(element_logits, element_atn, include_min=True)
        loss_mil_orig, _ = self.topkloss(element_logits,
                                         labels,
                                         is_back=True,
                                         rat=args['opt'].k,
                                         reduce=None)
        # SAL
        loss_mil_supp, _ = self.topkloss(element_logits_supp,
                                         labels,
                                         is_back=False,
                                         rat=args['opt'].k,
                                         reduce=None)

        edl_loss = self.edl_loss(element_logits_supp,
                                 element_atn,
                                 labels,
                                 rat=args['opt'].rat_atn,
                                 n_class=args['opt'].num_class,
                                 epoch=args['itr'],
                                 total_epoch=args['opt'].max_iter,
                                 )

        uct_guide_loss = self.uct_guide_loss(element_logits,
                                             element_logits_supp,
                                             element_atn,
                                             v_atn,
                                             f_atn,
                                             n_class=args['opt'].num_class,
                                             epoch=args['itr'],
                                             total_epoch=args['opt'].max_iter,
                                             amplitude=args['opt'].amplitude,
                                             )

        loss_3_supp_Contrastive = self.Contrastive(feat, element_logits_supp, labels, is_back=False)

        loss_norm = element_atn.mean()
        # guide loss
        loss_guide = (1 - element_atn -
                      element_logits.softmax(-1)[..., [-1]]).abs().mean()

        v_loss_norm = v_atn.mean()
        # guide loss
        v_loss_guide = (1 - v_atn -
                        element_logits.softmax(-1)[..., [-1]]).abs().mean()

        f_loss_norm = f_atn.mean()
        # guide loss
        f_loss_guide = (1 - f_atn -
                        element_logits.softmax(-1)[..., [-1]]).abs().mean()

        # total loss
        total_loss = (
                    args['opt'].alpha_edl * edl_loss +
                    args['opt'].alpha_uct_guide * uct_guide_loss +
                    loss_mil_orig.mean() + loss_mil_supp.mean() +
                    args['opt'].alpha3 * loss_3_supp_Contrastive +
                    args['opt'].alpha4 * mutual_loss +
                    args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3 +
                    args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3)

        loss_dict = {
            'edl_loss': args['opt'].alpha_edl * edl_loss,
            'uct_guide_loss': args['opt'].alpha_uct_guide * uct_guide_loss,
            'loss_mil_orig': loss_mil_orig.mean(),
            'loss_mil_supp': loss_mil_supp.mean(),
            'loss_supp_contrastive': args['opt'].alpha3 * loss_3_supp_Contrastive,
            'mutual_loss': args['opt'].alpha4 * mutual_loss,
            'norm_loss': args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3,
            'guide_loss': args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3,
            'total_loss': total_loss,
        }

        return total_loss, loss_dict

    def uct_guide_loss(self,
                       element_logits,
                       element_logits_supp,
                       element_atn,
                       v_atn,
                       f_atn,
                       n_class,
                       epoch,
                       total_epoch,
                       amplitude):

        evidence = exp_evidence(element_logits_supp)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=-1)
        snippet_uct = n_class / S

        total_snippet_num = element_logits.shape[1]
        curve = self.course_function(epoch, total_epoch, total_snippet_num, amplitude).to(self.device)

        loss_guide = (1 - element_atn - element_logits.softmax(-1)[..., [-1]]).abs().squeeze()

        v_loss_guide = (1 - v_atn - element_logits.softmax(-1)[..., [-1]]).abs().squeeze()

        f_loss_guide = (1 - f_atn - element_logits.softmax(-1)[..., [-1]]).abs().squeeze()

        total_loss_guide = (loss_guide + v_loss_guide + f_loss_guide) / 3

        _, uct_indices = torch.sort(snippet_uct, dim=1)
        sorted_curve = torch.gather(curve.repeat(10, 1), 1, uct_indices)

        uct_guide_loss = torch.mul(sorted_curve, total_loss_guide).mean()

        return uct_guide_loss

    def edl_loss(self,
                 element_logits_supp,
                 element_atn,
                 labels,
                 rat,
                 n_class,
                 epoch=0,
                 total_epoch=5000,
                 ):

        k = max(1, int(element_logits_supp.shape[-2] // rat))

        atn_values, atn_idx = torch.topk(
            element_atn,
            k=k,
            dim=1
        )

        atn_idx_expand = atn_idx.expand([-1, -1, n_class + 1])
        topk_element_logits = torch.gather(element_logits_supp, 1, atn_idx_expand)[:, :, :-1]

        video_logits = topk_element_logits.mean(dim=1)

        edl_loss = EvidenceLoss(
            num_classes=n_class,
            evidence='relu',
            loss_type='mse',
            with_kldiv=False,
            with_avuloss=False,
            disentangle=False,
            annealing_method='exp')

        edl_results = edl_loss(
            output=video_logits,
            target=labels,
            epoch=epoch,
            total_epoch=total_epoch
        )

        edl_loss = edl_results['loss_cls'].mean()

        return edl_loss

    def course_function(self, epoch, total_epoch, total_snippet_num, amplitude):

        idx = torch.arange(total_snippet_num)

        # From -1 to 1
        theta = 2 * (idx + 0.5) / total_snippet_num - 1

        # From 1 to -1
        delta = - 2 * epoch / total_epoch + 1

        curve = amplitude * torch.tanh(theta * delta) + 1

        return curve

    def topkloss(self,
                 element_logits,
                 labels,
                 is_back=True,
                 lab_rand=None,
                 rat=8,
                 reduce=None):

        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        if lab_rand is not None:
            labels_with_back = torch.cat((labels, lab_rand), dim=-1)

        topk_val, topk_ind = torch.topk(
            element_logits,
            k=max(1, int(element_logits.shape[-2] // rat)),
            dim=-2)
        instance_logits = torch.mean(
            topk_val,
            dim=-2,
        )
        labels_with_back = labels_with_back / (
                torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)
        milloss = (-(labels_with_back *
                     F.log_softmax(instance_logits, dim=-1)).sum(dim=-1))
        if reduce is not None:
            milloss = milloss.mean()
        return milloss, topk_ind

    def Contrastive(self, x, element_logits, labels, is_back=False):
        if is_back:
            labels = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        sim_loss = 0.
        n_tmp = 0.
        _, n, c = element_logits.shape
        for i in range(0, 3 * 2, 2):
            atn1 = F.softmax(element_logits[i], dim=0)
            atn2 = F.softmax(element_logits[i + 1], dim=0)

            n1 = torch.FloatTensor([np.maximum(n - 1, 1)]).to(self.device)
            n2 = torch.FloatTensor([np.maximum(n - 1, 1)]).to(self.device)
            Hf1 = torch.mm(torch.transpose(x[i], 1, 0), atn1)  # (n_feature, n_class)
            Hf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), atn2)
            Lf1 = torch.mm(torch.transpose(x[i], 1, 0), (1 - atn1) / n1)
            Lf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), (1 - atn2) / n2)

            d1 = 1 - torch.sum(Hf1 * Hf2, dim=0) / (
                    torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))  # 1-similarity
            d2 = 1 - torch.sum(Hf1 * Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
            d3 = 1 - torch.sum(Hf2 * Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d2 + 0.5, torch.FloatTensor([0.]).to(self.device)) * labels[i, :] * labels[i + 1, :])
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d3 + 0.5, torch.FloatTensor([0.]).to(self.device)) * labels[i, :] * labels[i + 1, :])
            n_tmp = n_tmp + torch.sum(labels[i, :] * labels[i + 1, :])
        sim_loss = sim_loss / n_tmp
        return sim_loss

    def decompose(self, outputs, **args):
        feat, element_logits, atn_supp, atn_drop, element_atn = outputs

        return element_logits, element_atn

class AcceleratedAttentionPool1d(nn.Module):
    def __init__(self, hidden_dim, kernel):
        super(AcceleratedAttentionPool1d, self).__init__()
        self.kernel = kernel
        self.attention = CrossAttention(hidden_dim)

    def forward(self, x):
        batch_size, embed_dim, seq_len = x.size()

        # Calculate the padding needed for the input tensor
        padding = self.kernel // 2
        x_padded = F.pad(x, (padding, padding), mode='constant', value=0)

        # Initialize indices for efficient slicing
        indices = torch.arange(self.kernel).view(1, 1, -1).expand(batch_size, embed_dim, -1).to(x.device)

        # Apply CrossAttention with a step size of 1
        chunks = x_padded.unfold(2, self.kernel, 1)
        out_list = []

        for chunk in chunks.transpose(1, 2).contiguous().view(-1, embed_dim, self.kernel):
            out, _ = self.attention(chunk, x_padded, chunk)
            center_out = out.gather(dim=2, index=indices)
            out_list.append(center_out.view(batch_size, embed_dim, 1))

        # Stack and reshape the attention-weighted representations
        output = torch.cat(out_list, dim=2) / self.kernel

        return output

class AttentionPool1d(nn.Module):
    def __init__(self, hidden_dim, kernel, dropout_prob=0.0):
        super(AttentionPool1d, self).__init__()
        self.kernel = kernel
        self.attention = CrossAttention(hidden_dim, dropout_prob=0.0)

    def forward(self, x):
        if self.kernel == 1:
            return x
        batch_size, embed_dim, seq_len = x.size()

        # Calculate the padding needed for the input tensor
        padding = self.kernel // 2
        x_padded = F.pad(x, (padding, padding), mode='constant', value=0).transpose(-1,-2)

        # Initialize a list to store the outputs of CrossAttention
        output_list = []

        # Apply CrossAttention with a step size of 1
        for i in range(padding, x_padded.size(1)-padding):
            chunk = x_padded[:, i : i + 1, :]  # Get the current chunk
            windows = x_padded[:, i - padding: i + padding + 1, :]

            # Apply CrossAttention
            out, _ = self.attention(chunk, windows, windows)
            output_list.append(out)

        # Concatenate the attention-weighted representations
        output = torch.cat(output_list, dim=1).transpose(-1,-2)

        return output

class DELU_DDG(torch.nn.Module):
    def __init__(self, n_feature, n_class, **args):
        super().__init__()
        embed_dim = 2048
        dropout_ratio = args['opt'].dropout_ratio

        self.Attn = getattr(models, args['opt'].AWM)(1024, args)

        self.fusion = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, (1,), padding=0),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_ratio)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(embed_dim, embed_dim, (3,), padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.7),
            nn.Conv1d(embed_dim, n_class + 1, (1,))
        )

        _kernel = 13
        self.pool = nn.AvgPool1d(_kernel, 1, padding=_kernel // 2, count_include_pad=True) \
            if _kernel is not None else nn.Identity()

        self.apply(weights_init)

    def forward(self, inputs, is_training=True, **args):
        feat = inputs.transpose(-1, -2)
        v_atn, vfeat, f_atn, ffeat, loss = self.Attn(feat[:, :1024, :], feat[:, 1024:, :], is_training=is_training, **args)
        x_atn = (f_atn + v_atn) / 2
        tfeat = torch.cat((vfeat, ffeat), 1)
        nfeat = self.fusion(tfeat)
        x_cls = self.classifier(nfeat)

        if "ActivityNet" in args["opt"].dataset_name:
            x_cls = self.pool(x_cls)
            x_atn = self.pool(x_atn)
            f_atn = self.pool(f_atn)
            v_atn = self.pool(v_atn)

        outputs = {
            'feat': nfeat.transpose(-1, -2),
            'cas': x_cls.transpose(-1, -2),
            'attn': x_atn.transpose(-1, -2),
            'v_atn': v_atn.transpose(-1, -2),
            'f_atn': f_atn.transpose(-1, -2),
            'extra_loss': loss,
        }

        return outputs

    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def complementary_learning_loss(self, cas, labels):

        labels_with_back = torch.cat(
            (labels, torch.ones_like(labels[:, [0]])), dim=-1).unsqueeze(1)
        cas = F.softmax(cas, dim=-1)
        complementary_loss = torch.sum(-(1 - labels_with_back) * torch.log((1 - cas).clamp_(1e-6)), dim=-1)
        return complementary_loss.mean()

    def criterion(self, outputs, labels, **args):

        feat, element_logits, element_atn = outputs['feat'], outputs['cas'], outputs['attn']
        v_atn = outputs['v_atn']
        f_atn = outputs['f_atn']

        try:
            fc_loss = outputs['extra_loss']['feat_loss']
        except:
            fc_loss = 0

        mutual_loss = 0.5 * \
                      F.mse_loss(v_atn, f_atn.detach()) + 0.5 * \
                      F.mse_loss(f_atn, v_atn.detach())

        element_logits_supp = self._multiply(
            element_logits, element_atn, include_min=True)

        cl_loss = self.complementary_learning_loss(element_logits, labels)

        edl_loss = self.edl_loss(element_logits_supp,
                                 element_atn,
                                 labels,
                                 rat=args['opt'].rat_atn,
                                 n_class=args['opt'].num_class,
                                 epoch=args['itr'],
                                 total_epoch=args['opt'].max_iter,
                                 )

        uct_guide_loss = self.uct_guide_loss(element_logits,
                                             element_logits_supp,
                                             element_atn,
                                             v_atn,
                                             f_atn,
                                             n_class=args['opt'].num_class,
                                             epoch=args['itr'],
                                             total_epoch=args['opt'].max_iter,
                                             amplitude=args['opt'].amplitude,
                                             )

        loss_mil_orig, _ = self.topkloss(element_logits,
                                         labels,
                                         is_back=True,
                                         rat=args['opt'].k)

        # SAL
        loss_mil_supp, _ = self.topkloss(element_logits_supp,
                                         labels,
                                         is_back=False,
                                         rat=args['opt'].k)

        loss_3_supp_Contrastive = self.Contrastive(
            feat, element_logits_supp, labels, is_back=False)

        loss_norm = element_atn.mean()
        # guide loss
        loss_guide = (1 - element_atn -
                      element_logits.softmax(-1)[..., [-1]]).abs().mean()

        v_loss_norm = v_atn.mean()
        # guide loss
        v_loss_guide = (1 - v_atn -
                        element_logits.softmax(-1)[..., [-1]]).abs().mean()

        f_loss_norm = f_atn.mean()
        # guide loss
        f_loss_guide = (1 - f_atn -
                        element_logits.softmax(-1)[..., [-1]]).abs().mean()

        total_loss = (
                args['opt'].alpha_edl * edl_loss +
                args['opt'].alpha_uct_guide * uct_guide_loss +
                loss_mil_orig.mean() + loss_mil_supp.mean() +
                args['opt'].alpha3 * loss_3_supp_Contrastive +
                args['opt'].alpha4 * mutual_loss +
                args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3 +
                args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3
                + args['opt'].alpha5 * fc_loss
                + args['opt'].alpha6 * cl_loss
        )

        loss_dict = {
            'edl_loss': args['opt'].alpha_edl * edl_loss,
            'uct_guide_loss': args['opt'].alpha_uct_guide * uct_guide_loss,
            'loss_mil_orig': loss_mil_orig.mean(),
            'loss_mil_supp': loss_mil_supp.mean(),
            'loss_supp_contrastive': args['opt'].alpha3 * loss_3_supp_Contrastive,
            'mutual_loss': args['opt'].alpha4 * mutual_loss,
            'norm_loss': args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3,
            'guide_loss': args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3,
            'feat_loss': args['opt'].alpha5 * fc_loss,
            'complementary_loss': args['opt'].alpha6 * cl_loss,
            'total_loss': total_loss,
        }

        return total_loss, loss_dict

    def uct_guide_loss(self,
                       element_logits,
                       element_logits_supp,
                       element_atn,
                       v_atn,
                       f_atn,
                       n_class,
                       epoch,
                       total_epoch,
                       amplitude):

        evidence = exp_evidence(element_logits_supp)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=-1)
        snippet_uct = n_class / S

        total_snippet_num = element_logits.shape[1]
        curve = self.course_function(
            epoch, total_epoch, total_snippet_num, amplitude)

        loss_guide = (1 - element_atn - element_logits.softmax(-1)
        [..., [-1]]).abs().squeeze()

        v_loss_guide = (1 - v_atn - element_logits.softmax(-1)
        [..., [-1]]).abs().squeeze()

        f_loss_guide = (1 - f_atn - element_logits.softmax(-1)
        [..., [-1]]).abs().squeeze()

        total_loss_guide = (loss_guide + v_loss_guide + f_loss_guide) / 3

        _, uct_indices = torch.sort(snippet_uct, dim=1)
        sorted_curve = torch.gather(curve.repeat(10, 1), 1, uct_indices)

        uct_guide_loss = torch.mul(sorted_curve, total_loss_guide).mean()

        return uct_guide_loss

    def edl_loss(self,
                 element_logits_supp,
                 element_atn,
                 labels,
                 rat,
                 n_class,
                 epoch=0,
                 total_epoch=5000,
                 ):

        k = max(1, int(element_logits_supp.shape[-2] // rat))

        atn_values, atn_idx = torch.topk(
            element_atn,
            k=k,
            dim=1
        )
        atn_idx_expand = atn_idx.expand([-1, -1, n_class + 1])
        topk_element_logits = torch.gather(
            element_logits_supp, 1, atn_idx_expand)[:, :, :-1]
        video_logits = topk_element_logits.mean(dim=1)

        edl_loss = EvidenceLoss(
            num_classes=n_class,
            evidence='exp',
            loss_type='log',
            with_kldiv=False,
            with_avuloss=False,
            disentangle=False,
            annealing_method='exp')

        edl_results = edl_loss(
            output=video_logits,
            target=labels,
            epoch=epoch,
            total_epoch=total_epoch
        )

        edl_loss = edl_results['loss_cls'].mean()

        return edl_loss

    def course_function(self, epoch, total_epoch, total_snippet_num, amplitude):

        idx = torch.arange(total_snippet_num)
        theta = 2 * (idx + 0.5) / total_snippet_num - 1
        delta = - 2 * epoch / total_epoch + 1
        curve = amplitude * torch.tanh(theta * delta) + 1

        return curve

    def topkloss(self,
                 element_logits,
                 labels,
                 is_back=True,
                 rat=8):

        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)

        topk_val, topk_ind = torch.topk(
            element_logits,
            k=max(1, int(element_logits.shape[-2] // rat)),
            dim=-2)

        instance_logits = torch.mean(topk_val, dim=-2)

        labels_with_back = labels_with_back / (
                torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)

        milloss = - (labels_with_back *
                     F.log_softmax(instance_logits, dim=-1)).sum(dim=-1)

        return milloss, topk_ind

    def Contrastive(self, x, element_logits, labels, is_back=False):
        if is_back:
            labels = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        sim_loss = 0.
        n_tmp = 0.
        _, n, c = element_logits.shape
        for i in range(0, 3 * 2, 2):
            atn1 = F.softmax(element_logits[i], dim=0)
            atn2 = F.softmax(element_logits[i + 1], dim=0)

            n1 = torch.FloatTensor([np.maximum(n - 1, 1)]).cuda()
            n2 = torch.FloatTensor([np.maximum(n - 1, 1)]).cuda()
            # (n_feature, n_class)
            Hf1 = torch.mm(torch.transpose(x[i], 1, 0), atn1)
            Hf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), atn2)
            Lf1 = torch.mm(torch.transpose(x[i], 1, 0), (1 - atn1) / n1)
            Lf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), (1 - atn2) / n2)

            d1 = 1 - torch.sum(Hf1 * Hf2, dim=0) / (
                    torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))  # 1-similarity
            d2 = 1 - torch.sum(Hf1 * Lf2, dim=0) / \
                 (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
            d3 = 1 - torch.sum(Hf2 * Lf1, dim=0) / \
                 (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d2 + 0.5, torch.FloatTensor([0.]).cuda()) * labels[i, :] * labels[i + 1, :])
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d3 + 0.5, torch.FloatTensor([0.]).cuda()) * labels[i, :] * labels[i + 1, :])
            n_tmp = n_tmp + torch.sum(labels[i, :] * labels[i + 1, :])
        sim_loss = sim_loss / n_tmp
        return sim_loss

    def decompose(self, outputs, **args):
        feat, element_logits, atn_supp, atn_drop, element_atn = outputs

        return element_logits, element_atn

class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, dropout_prob=0.0, temp=1.0):
        super(CrossAttention, self).__init__()

        temp = 1.5 # 0.01 # 2-2快的是0.1 1-1慢的是0.5

        self.hidden_dim = hidden_dim
        self.temp = temp

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

        self.register_buffer("scale", torch.sqrt(torch.FloatTensor([hidden_dim])))

        # Add dropout layer with the specified dropout probability
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, input_q, input_k, input_v):
        Q = self.query(input_q)
        K = self.query(input_k)
        V = input_v

        energy = torch.matmul(Q, K.permute(0, 2, 1)) / self.scale

        attention = torch.softmax(energy/self.temp, dim=-1)

        # print(attention)

        # print(attention[0,0,6], torch.max(attention[0]))

        # Apply dropout to the attention scores
        # attention = self.dropout(attention)

        out = torch.matmul(attention, V)
        out = out.contiguous()

        out = self.fc_out(out)

        return out, attention

class NewSubclassBranch(nn.Module):
    def __init__(self, num_sub_class=50, num_class=20):
        super(NewSubclassBranch, self).__init__()

        self.embed_dim = 2048
        self.num_sub_class_per_class = num_sub_class
        self.num_sub_class = num_sub_class # YZ: num_sub_class changed to the number of sub-classes in each class
        self.num_class = num_class

        self.register_buffer("sub_classtype", torch.randn((self.num_sub_class, self.embed_dim)))
        self.register_buffer("class_type", torch.randn((self.num_class+1, self.embed_dim)))
        self.register_buffer("fb_type", torch.randn((2, self.embed_dim)))

        self.snippet_to_sub_mapping = CrossAttention(self.embed_dim, True) # 从snippet到子类的similarity matrix 以及subclass prototype的重表征
        self.sub_to_fg_mapping = CrossAttention(self.embed_dim, False) # 从sub_class到前后景, 且只取2分类结果, 这里用主分支和otsu进行约束
        self.sub_to_class_mapping = CrossAttention(self.embed_dim, False) # 从sub_classc到前景class
    
    def update(self):
        pass

    def forward(self, snippets):
        '''
        snippets: (b, c, t)
        '''
        b, _, t = snippets.size()
        snippets_feature = rearrange(snippets, 'b c t -> (b t) c')

        sub_class_aggregation_feature, snip2sub_sim_matrix = self.snippet_to_sub_mapping(snippets_feature, self.sub_classtype, self.sub_classtype)  # map to sub_class space
        sub2fg_sim_matrix = self.sub_to_fg_mapping(sub_class_aggregation_feature, self.fb_type)
        sub2class_sim_matrix = self.sub_to_class_mapping(sub_class_aggregation_feature, self.class_type)

        snip2sub_sim_matrix = rearrange(snip2sub_sim_matrix, '(b t) s -> b t s', b=b)
        sub2fg_sim_matrix = rearrange(sub2fg_sim_matrix, '(b t) e -> b t e', b=b)[:,:,0:1]
        sub2class_sim_matrix = rearrange(sub2class_sim_matrix, '(b t) c -> b t c', b=b)

        return snip2sub_sim_matrix, sub2fg_sim_matrix, sub2class_sim_matrix  # Return the processed snippets

class InverseAvgPool1d(nn.Module):
    def __init__(self, kernel_size):
        super(InverseAvgPool1d, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        # Pad the tenasor along the last dimension
        # x_padded = F.pad(x, (self.kernel_size // 2, self.kernel_size // 2))
        left_pad = x[:, :, :1]  # Take the first element along the last dimension
        right_pad = x[:, :, -1:]  # Take the last element along the last dimension
        x_padded = F.pad(x, (self.kernel_size // 2, self.kernel_size // 2), 'constant', value=0)  # Pad with zeros
        x_padded[:, :, :self.kernel_size // 2] = left_pad  # Replace left padding with x[0]
        x_padded[:, :, -self.kernel_size // 2:] = right_pad  # Replace right padding with x[-1]

        # Create an empty tensor with the same size as x
        inverse = torch.zeros_like(x_padded)

        # Handle the first element separately
        inverse[..., 0] = self.kernel_size * x[..., 0]

        # Iterate over the tensor along the dimension of size kernel_size
        for i in range(x.size(-1) - 1):
            inverse[..., i + self.kernel_size // 2 + 1] = inverse[..., i - self.kernel_size // 2] + self.kernel_size * (x_padded[..., i + 1] - x_padded[..., i])

        # Remove padding
        inverse = inverse[..., self.kernel_size // 2: -self.kernel_size // 2]

        return inverse
    
class DirtyAvgPool(nn.Module):
        def __init__(self, scale):
            super(DirtyAvgPool, self).__init__()
            self.scale = scale
            self.padding = nn.ZeroPad2d((self.scale//2, self.scale//2, 0, 0))

        def forward(self, x):
            return x
            if self.scale == 1:
                return x

            b, e, t = x.size()
            y = x
            # print(y.size())
            x = self.padding(x)
            # print(x.size())
            for i in range(self.scale//2, x.size(-1)-self.scale//2):
                start = i - self.scale//2
                end = i + self.scale//2+1
                window = x[:,:,start:end]

                # print(window.size(), self.scale)
                center=x[:,:,i]
                left = window[:,:,:self.scale//2].max(dim=-1)[0]
                right = window[:,:,-self.scale//2:].max(dim=-1)[0]
                replace = torch.min(left, right)
                mask = center < replace
                # center[mask] = center[mask] + (replace[mask] - center[mask]) * 0.8
                center[mask] = center[mask] - (replace[mask] - center[mask]) * 0.3
                y[:,:,i-self.scale//2] = center
            return y
