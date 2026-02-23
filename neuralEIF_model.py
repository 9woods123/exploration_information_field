import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class FourierEncoding(nn.Module):
    def __init__(self, input_dim=2, num_freqs=64, scale=5.0):
        super().__init__()
        B = torch.randn(input_dim, num_freqs) * scale
        self.register_buffer("B", B)

    def forward(self, x):
        # x: (N,2)
        proj = 2 * np.pi * x @ self.B
        return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)


class NeuralEIFField(nn.Module):
    """
    Learn I(x,y)
    """

    def __init__(self, hidden_dim=128, num_layers=4, num_freqs=64):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.encoder = FourierEncoding(2, num_freqs)
        in_dim = 2 + 2 * num_freqs

        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = x / 20.0  # normalize to [-1,1] approx
        feat = self.encoder(x)
        return self.mlp(feat)

    # -------------------------
    # training
    # -------------------------
    def fit(
        self,
        ts,
        Is,
        epochs=1000,
        lr=1e-3,
        weight_decay=1e-4,
        lambda_grad=1e-2,
        smooth_samples=512,
        map_bound=10.0
    ):

        self.to(self.device)
        self.train()

        X = torch.tensor(ts, dtype=torch.float32).to(self.device)
        Y = torch.tensor(Is, dtype=torch.float32).unsqueeze(1).to(self.device)

        optimizer = optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        mse_loss = nn.MSELoss()

        for ep in range(epochs):

            # -------------------------
            # Supervised MSE
            # -------------------------
            pred = self.forward(X)
            loss_data = mse_loss(pred, Y)

            # -------------------------
            # Smoothness Regularization
            # -------------------------
            xs = (torch.rand(smooth_samples, 2) * 2 - 1) * map_bound
            xs = xs.to(self.device)
            xs.requires_grad_(True)

            ys = self.forward(xs)

            grads = torch.autograd.grad(
                ys.sum(),
                xs,
                create_graph=True
            )[0]

            loss_smooth = (grads.norm(dim=-1) ** 2).mean()

            # -------------------------
            # Total Loss
            # -------------------------
            loss = loss_data + lambda_grad * loss_smooth

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ep % 50 == 0:
                print(f"[Neural EIF] Ep {ep} | "
                    f"MSE: {loss_data.item():.6f} | "
                    f"Smooth: {loss_smooth.item():.6f}")

        print("Neural EIF training done.")

    # -------------------------
    # eval single point
    # -------------------------
    def query(self, t):
        super().eval()   # 切换到 inference 模式

        with torch.no_grad():
            x = torch.tensor(t, dtype=torch.float32).unsqueeze(0).to(self.device)
            return self.forward(x).item()


    # -------------------------
    # gradient
    # -------------------------
    def grad(self, t):
        super().eval()

        x = torch.tensor(t, dtype=torch.float32).unsqueeze(0).to(self.device)
        x.requires_grad_(True)
        y = self.forward(x)
        y.backward()
        return x.grad.detach().cpu().numpy()[0]
    



class GaussianNeuralEIF(nn.Module):
    def __init__(self, K=50, init_scale=5.0, device='cpu'):
        super().__init__()
        self.K = K
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # centers μ_k
        self.mu = nn.Parameter(
            torch.randn(K, 2) * init_scale
        )

        # log σ_k  (保证正数)
        self.log_sigma = nn.Parameter(
            torch.zeros(K)  # 初始 σ=1
        )

        # weights
        self.w = nn.Parameter(
            torch.randn(K) * 0.1
        )

        self.to(device)

    # ---------------------------------
    # forward
    # ---------------------------------
    def forward(self, x):
        """
        x: (N,2)
        return: (N,1)
        """
        x = x.unsqueeze(1)                 # (N,1,2)
        mu = self.mu.unsqueeze(0)         # (1,K,2)

        diff = x - mu                     # (N,K,2)

        sigma = torch.exp(self.log_sigma) # (K,)
        sigma = sigma.unsqueeze(0)        # (1,K)

        dist2 = (diff ** 2).sum(dim=-1)   # (N,K)

        gauss = torch.exp(-0.5 * dist2 / (sigma ** 2))  # (N,K)

        out = (gauss * self.w.unsqueeze(0)).sum(dim=1)

        return out.unsqueeze(-1)

    # ---------------------------------
    # fit
    # ---------------------------------
    def fit(self,
            xs,
            ys,
            epochs=2000,
            lr=1e-3,
            weight_decay=0.0,
            lambda_l1=0.0,
            verbose=True):
        
        self.to(self.device)

        self.train()

        xs = torch.tensor(xs, dtype=torch.float32).to(self.device)
        ys = torch.tensor(ys, dtype=torch.float32).to(self.device)

        optimizer = optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        for e in range(epochs):

            optimizer.zero_grad()

            pred = self.forward(xs)

            loss_data = ((pred - ys) ** 2).mean()

            # 稀疏正则（可选）
            loss_sparse = lambda_l1 * self.w.abs().sum()

            loss = loss_data + loss_sparse

            loss.backward()
            optimizer.step()

            if verbose and e % 200 == 0:
                print(f"Epoch {e} | Loss: {loss.item():.6f}")

        self.eval()

    # ---------------------------------
    # query (numpy接口)
    # ---------------------------------
    def query(self, x):
        """
        x: (2,) or (N,2)
        return numpy
        """
        super().eval()   # 切换到 inference 模式

        if isinstance(x, np.ndarray):
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        if x_tensor.dim() == 1:
            x_tensor = x_tensor.unsqueeze(0)

        with torch.no_grad():
            y = self.forward(x_tensor)

        return y.cpu().numpy()

    # ---------------------------------
    # analytic gradient (更稳定)
    # ---------------------------------
    def grad(self, x):
        """
        x: (2,) numpy
        return: (2,) numpy
        """

        super().eval()   # 切换到 inference 模式

        x = torch.tensor(x, dtype=torch.float32).to(self.device)

        mu = self.mu                        # (K,2)
        sigma = torch.exp(self.log_sigma)   # (K,)
        w = self.w                          # (K,)

        diff = x.unsqueeze(0) - mu          # (K,2)

        dist2 = (diff ** 2).sum(dim=-1)     # (K,)

        gauss = torch.exp(-0.5 * dist2 / (sigma ** 2))  # (K,)

        # ∇I(x) = Σ w_k * exp(...) * (-(x-μ_k)/σ_k^2)
        grad = (
            w.unsqueeze(-1)
            * gauss.unsqueeze(-1)
            * (-diff / (sigma.unsqueeze(-1) ** 2))
        ).sum(dim=0)

        return grad.detach().cpu().numpy()