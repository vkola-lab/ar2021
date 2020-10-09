import torch


def dice_coefficient(x, y, n_channels):
    dice = []
    x = torch.argmax(x, 1)
    for c in range(1, n_channels):
        tp = (((x == c) & (y == c)).sum().item())
        div = ((x == c).sum().item() + (y == c).sum().item())
        dice.append(2 * tp / div)

    return dice

