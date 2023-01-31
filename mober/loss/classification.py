import torch

from torch.nn import NLLLoss


def loss_function_classification(pred, target, class_weights):
    """
    Compute negative log likelihood loss.

    :param pred: predictions
    :param target: actual classes
    :param class_weights: weights - one per class
    :return: Weighted prediction loss. Summed for all the samples, not averaged.
    """
    loss_function = NLLLoss(weight=class_weights, reduction="none")
    return loss_function(pred, torch.argmax(target, dim=1)).sum(dim=0)
