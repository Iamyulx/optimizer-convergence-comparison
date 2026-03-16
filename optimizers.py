import torch


class SGD_Scratch:

    def __init__(self, params, lr=1e-2):

        self.params = list(params)
        self.lr = lr

    def step(self):

        with torch.no_grad():

            for p in self.params:

                p -= self.lr * p.grad

    def zero_grad(self):

        for p in self.params:

            p.grad = None


class Adam_Scratch:

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):

        self.params = list(params)

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0

        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):

        self.t += 1

        with torch.no_grad():

            for i, p in enumerate(self.params):

                g = p.grad

                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):

        for p in self.params:

            p.grad = None


class AdamW_Scratch:

    def __init__(self, params, lr=1e-3, weight_decay=1e-2,
                 beta1=0.9, beta2=0.999, eps=1e-8):

        self.params = list(params)

        self.lr = lr
        self.weight_decay = weight_decay

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0

        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):

        self.t += 1

        with torch.no_grad():

            for i, p in enumerate(self.params):

                g = p.grad

                # decoupled weight decay
                p *= (1 - self.lr * self.weight_decay)

                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):

        for p in self.params:

            p.grad = None
