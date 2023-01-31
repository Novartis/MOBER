import torch

from torch import optim, nn


def create_model(model_cls, device, *args, filename=None, lr=1e-3, **kwargs):
    """
    Simple model serialization to resume training from given epoch.

    :param model_cls: Model definition
    :param device: Device (cpu or gpu)
    :param args: arguments to be passed to the model constructor
    :param filename: filename if the model is to be loaded
    :param lr: learning rate to be used by the model optimizer
    :param kwargs: keyword arguments to be used by the model constructor
    :return:
    """
    model = model_cls(*args, **kwargs)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if filename is not None:
        checkpoint = torch.load(filename, map_location=torch.device("cpu"))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.load_state_dict(checkpoint["model_state_dict"])

        print(f"Loaded model epoch: {checkpoint['epoch']}, loss {checkpoint['loss']}")

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print("Loading model on ", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    return model.to(device), optimizer


def save_model(model, optimizer, epoch, loss, filename, device):
    """
    Save the model to a file.

    :param model: model to be saved
    :param optimizer: model optimizer
    :param epoch: number of epoch, only for information
    :param loss: loss, only for information
    :param filename: where to save the model
    :param device: device of the model
    """
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    torch.save({
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }, filename)
