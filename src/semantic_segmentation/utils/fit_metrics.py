import numpy as np
import torch.nn.functional as F
import torch
from src.semantic_segmentation.loaders.dataloader import labels_classes


def dice_loss(input, target):
    # performs softmax over last layer before computing loss
    # backprop can be called using dice_loss.backward()
    # input: final layer output before softmax (nxcxwxh)
    # target: one-hot encoded labels (nxcxwxh)

    assert input.size() == target.size()

    ####################################

    # prod_p_g = pred * target
    # prod_p_g = torch.sum(prod_p_g, dim = (1, 2, 3))

    # cardinality = torch.sum(pred + target, dim = (1, 2, 3))

    # score = 2. * prod_p_g / (cardinality + 0.00001)

    # return torch.mean(torch.tensor(1.) - score)

    ####################################

    pred = F.softmax(input, dim=1)

    p_g = pred * target
    p_g = torch.sum(torch.sum(p_g, dim=3), dim=2)

    p_sq = pred * pred
    p_sq = torch.sum(torch.sum(p_sq, dim=3), dim=2)

    g_sq = target * target
    g_sq = torch.sum(torch.sum(g_sq, dim=3), dim=2)

    dice = 2. * (p_g / (p_sq + g_sq + 0.00001))
    dice_per_channel = 1. - (torch.sum(dice, dim=0))

    dice_total = torch.sum(dice_per_channel) / dice_per_channel.size(0)

    return dice_total


def iou(pred, labels):
    # This function returns a list that contains the iou for each class over a batch of images

    ious = []
    n_class = pred.shape[1]

    pred = torch.argmax(pred, dim=1)

    pred = pred.view(pred.shape[0], -1)
    labels = labels.view(labels.shape[0], -1)

    for cls in range(n_class):
        if labels_classes[cls].trainId == 255:
            continue

        intersection = torch.sum(torch.mul((pred == cls).cuda(), (labels == cls).cuda()))
        intersection_fl = intersection.item()
        union = torch.sum(pred == cls) + torch.sum(labels == cls) - intersection
        union_fl = union.item()
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection_fl) / float(union_fl))
    return ious


def pixel_acc(pred, labels):
    pred = torch.argmax(pred, dim=1)

    acc = float(torch.sum(pred == labels)) / (pred.shape[0] * pred.shape[1] * pred.shape[2])
    return acc * 100
