import torch.optim as optim


def create_optimizer(model, lr: float = 1e-3, weight_decay: float = 1e-5) -> optim.Adam:
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
