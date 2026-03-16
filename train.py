import torch
import torch.nn.functional as F

from model import LinearModel
from optimizers import SGD_Scratch, Adam_Scratch, AdamW_Scratch


torch.manual_seed(42)

N = 1000

X = torch.randn(N, 10)

true_w = torch.randn(10, 1)

y = X @ true_w + 0.1 * torch.randn(N, 1)


def train(optimizer_class, epochs=200, lr=1e-3):

    model = LinearModel()

    optimizer = optimizer_class(model.parameters(), lr=lr)

    losses = []

    for step in range(epochs):

        pred = model(X)

        loss = F.mse_loss(pred, y)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        losses.append(loss.item())

    return losses
