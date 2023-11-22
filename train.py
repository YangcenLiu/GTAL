import torch
import numpy as np
import torch.nn as nn
from torch.autograd import grad

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def random_resize(feature, min_ratio: float, max_ratio: float):
    assert 0 < min_ratio <= max_ratio
    ratio = random.uniform(min_ratio, max_ratio)

    b, t, c = feature.shape
    feature = feature.permute(0, 2, 1)
    new_t = int(t * ratio)

    resized_feature = F.interpolate(
        feature, size=new_t, mode="linear", align_corners=False
    )
    resized_feature = resized_feature.permute(0, 2, 1)
    return resized_feature

def train(itr, dataset, args, model, optimizer, device):
    model.train()
    features, labels, pairs_id = dataset.load_data(n_similar=args.num_similar)
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:, :np.max(seq_len), :]

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    # if args.random_resize:
        # features = random_resize(features, args.resize_min_ratio, args.resize_max_ratio)

    outputs = model(features, seq_len=seq_len, is_training=True, itr=itr, opt=args)
    total_loss, loss_dict = model.criterion(outputs, labels, seq_len=seq_len, device=device, opt=args,
                                            itr=itr, pairs_id=pairs_id, inputs=features)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.data.cpu().numpy()

def env(features, env, device):
    env_bias = torch.rand_like(features)*2-1
    env_bias *= env

    return features + env_bias.to(device)

def compute_irm_penalty(losses1, losses2, dummy_w):
    g1 = grad(losses1, dummy_w, create_graph=True)[0]
    g2 = grad(losses2, dummy_w, create_graph=True)[0]
    irm_penalty = (g1 * g2).mean()
    return irm_penalty

def train_irm(itr, dataset, args, model, optimizer, device):
    model.train()
    features1, labels, pairs_id, features2 = dataset.load_data(n_similar=args.num_similar, return_env=True)
    
    seq_len = np.sum(np.max(np.abs(features1), axis=2) > 0, axis=1)
    features1 = features1[:, :np.max(seq_len), :]

    features1 = torch.from_numpy(features1).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    features2 = features2[:, :np.max(seq_len), :]

    features2 = torch.from_numpy(features2).float().to(device)

    outputs1 = model(features1, is_training=True)
    outputs2 = model(features2, is_training=True)

    total_loss1, loss_dict1 = model.criterion(outputs1, labels, seq_len=seq_len, device=device, opt=args,
                                            itr=itr, pairs_id=pairs_id, inputs=features1)
    
    total_loss2, loss_dict2 = model.criterion(outputs2, labels, seq_len=seq_len, device=device, opt=args,
                                            itr=itr, pairs_id=pairs_id, inputs=features2)

    dummy_w = model.classifier[1].weight
    irm_p = compute_irm_penalty(total_loss1, total_loss2, dummy_w)

    total_loss1 = total_loss1 + irm_p # 0.05*irm_p

    optimizer.zero_grad()
    total_loss1.backward()
    optimizer.step()

    return total_loss1.data.cpu().numpy()

def train_adapter(itr, dataset, args, model, optimizer, device):
    model.eval()
    total_loss = 0

    bs = vars(args)["batch_size"]

    for i in range(bs):
        features, labels, pairs_id = dataset.load_data(n_similar=args.num_similar)

        features = torch.from_numpy(features[i]).float().to(device).unsqueeze(0)
        labels = torch.from_numpy(labels).float().to(device)

        outputs = model.adapter_update(features)
        single_loss, loss_dict = model.criterion(outputs, labels, device=device, opt=args,
                                                itr=itr, pairs_id=pairs_id, inputs=features)
        total_loss += single_loss
    
    total_loss = total_loss / bs

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    update_param = 0.9 # 0.9
    model.EMA_update(update_param)

    return total_loss.data.cpu().numpy()

def ddp_train(itr, dataset, args, model, optimizer, device):
    model = model.module
    model.train()
    features, labels, pairs_id = dataset.load_data(n_similar=args.num_similar)
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:, :np.max(seq_len), :]

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    outputs = model(features, seq_len=seq_len, is_training=True, itr=itr, opt=args)
    total_loss, loss_dict = model.criterion(outputs, labels, seq_len=seq_len, device=device, opt=args,
                                            itr=itr, pairs_id=pairs_id, inputs=features)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.data.cpu().numpy()

def fgsm_attack(feature, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = feature + epsilon * sign_data_grad
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def train_snip(itr, dataset, args, model, optimizer, writer=None):
    model.train()
    features, labels, pairs_id = dataset.load_data(n_similar=args.num_similar)
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:, : np.max(seq_len), :]
    if args.modality == "rgb":
        features = features[:, :, :1024]
    elif args.modality == "flow":
        features = features[:, :, 1024:]
    elif args.modality == "fusion":
        pass
    else:
        raise NotImplementedError

    features = torch.from_numpy(features).float().to(args.device)  # b, t, c
    labels = torch.from_numpy(labels).float().to(args.device)

    if args.random_resize:
        features = random_resize(features, args.resize_min_ratio, args.resize_max_ratio)

    if args.adv_attack:
        features.requires_grad = True
    outputs = model(features, seq_len=seq_len, is_training=True, itr=itr, opt=args)
    total_loss, loss_dict = model.criterion(
        outputs,
        labels,
        seq_len=seq_len,
        device=args.device,
        opt=args,
        itr=itr,
        pairs_id=pairs_id,
        inputs=features,
    )

    if args.adv_attack:
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        data_grad = features.grad.data
        pert_feature = fgsm_attack(features, args.epsilon, data_grad)
        pert_outputs = model(
            pert_feature, seq_len=seq_len, is_training=True, itr=itr, opt=args
        )
        pert_total_loss, pert_loss_dict = model.criterion(
            pert_outputs,
            labels,
            seq_len=seq_len,
            device=args.device,
            opt=args,
            itr=itr,
            pairs_id=pairs_id,
            inputs=features,
        )

        total_loss = total_loss + pert_total_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if writer is not None:
        for k, v in loss_dict.items():
            writer.add_scalar("train/{}".format(k), loss_dict[k].item(), itr)

    return total_loss.data.cpu().numpy()


# PGD attack code
def pgd_attack(model, images, labels, device, epsilon, num_iter, args):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()

    ori_images = images.clone().detach()
    for i in range(num_iter):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost_mil_orig = model.topkloss(
            outputs["cas"], labels, is_back=True, rat=args.k
        )[0].mean()
        element_logits_supp = model._multiply(
            outputs["cas"], outputs["attn"], include_min=True
        )
        cost_mil_supp = model.topkloss(
            element_logits_supp, labels, is_back=False, rat=args.k
        )[0].mean()
        cost = cost_mil_orig + cost_mil_supp
        cost.backward()

        attack_images = images + epsilon * images.grad.sign()
        eta = attack_images - ori_images
        images = (ori_images + eta).detach()

    return images


def train_w_pgd(itr, dataset, args, model, optimizer, device):
    model.train()
    features, labels, pairs_id = dataset.load_data(n_similar=args.num_similar)
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:, : np.max(seq_len), :]

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    pert_features = pgd_attack(
        model, features, labels, device, args.epsilon, args.num_pgd_iter, args
    )

    optimizer.zero_grad()
    outputs = model(pert_features, seq_len=seq_len, is_training=True, itr=itr, opt=args)
    total_loss, loss_dict = model.criterion(
        outputs,
        labels,
        seq_len=seq_len,
        device=device,
        opt=args,
        itr=itr,
        pairs_id=pairs_id,
        inputs=features,
    )

    total_loss.backward()
    optimizer.step()

    return total_loss.data.cpu().numpy()
