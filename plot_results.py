import matplotlib.pyplot as plt
import torch

from train import train
from optimizers import SGD_Scratch, Adam_Scratch, AdamW_Scratch


def train_torch(optimizer_class, epochs=200, lr=1e-3):

    import torch.nn.functional as F
    from model import LinearModel

    model = LinearModel()

    optimizer = optimizer_class(model.parameters(), lr=lr)

    losses = []

    for step in range(epochs):

        pred = model(X)

        loss = F.mse_loss(pred, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        losses.append(loss.item())

    return losses


loss_sgd = train(SGD_Scratch, lr=1e-2)
loss_adam = train(Adam_Scratch, lr=1e-3)
loss_adamw = train(AdamW_Scratch, lr=1e-3)

loss_adam_torch = train_torch(torch.optim.Adam, lr=1e-3)
loss_adamw_torch = train_torch(torch.optim.AdamW, lr=1e-3)


plt.figure(figsize=(8,6))

plt.plot(loss_sgd, label="SGD (scratch)")
plt.plot(loss_adam, label="Adam (scratch)")
plt.plot(loss_adamw, label="AdamW (scratch)")

plt.plot(loss_adam_torch, "--", label="Adam (torch)")
plt.plot(loss_adamw_torch, "--", label="AdamW (torch)")

plt.xlabel("Training Steps")
plt.ylabel("MSE Loss")
plt.title("Optimizer Convergence Comparison")

plt.legend()
plt.grid(True)

plt.savefig("convergence_plot.png")

plt.show()
