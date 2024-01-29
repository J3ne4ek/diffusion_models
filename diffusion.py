import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from enum import Enum


class TargetType(Enum):
    NOISE = 1
    X_START = 2


class DiffusionModel:
    def __init__(self, model, img_size, n_steps=1000, model_type=TargetType.NOISE, is_conditioned=False,
                 min_beta=1e-4, max_beta=0.02):
        self.n_steps = n_steps
        self.img_size = img_size
        self.type = model_type
        self.is_conditioned = is_conditioned
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(self.device)
        self.model = model
        self.alphas = 1 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)
        self.prev_alpha_hats = F.pad(self.alpha_hats[:-1], (1, 0), value=1.)
        self.beta_hats = self.betas * (1. - self.prev_alpha_hats) / (1. - self.alpha_hats)

    def get_statistics(self, x_start, x_t, t):
        var = self.beta_hats[t][:, None, None, None]
        alpha_hat = self.alpha_hats[t][:, None, None, None]
        prev_alphas_hat = self.prev_alpha_hats[t][:, None, None, None]
        beta_hat = self.beta_hats[t][:, None, None, None]

        b = (x_t - alpha_hat.sqrt() * x_start) / (1. - alpha_hat).sqrt()
        mean = prev_alphas_hat.sqrt() * x_start + (1. - prev_alphas_hat - beta_hat).sqrt() * b
        return mean, torch.log(var.clamp(min=1e-20))

    def forward_process(self, x, t, noise):
        a_hats = self.alpha_hats[t].reshape(len(x), 1, 1, 1)
        return a_hats.sqrt() * x + (1 - a_hats).sqrt() * noise

    def sample_prev(self, x_t, t, i, w, labels):
        if self.type == TargetType.NOISE:
            predicted_noise = self.model(x_t, t)
            if (w > 0) and (labels is not None):
                predicted_noise2 = self.model(x_t, t, labels)
                predicted_noise = (w + 1) * predicted_noise2 - w * predicted_noise

            alpha = self.alphas[i]
            alpha_hat = self.alpha_hats[i]
            beta = self.betas[i]
            noise = torch.randn_like(x_t) if i > 1 else 0

            return 1 / torch.sqrt(alpha) * (
                        x_t - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        else:
            x_start = self.model(x_t, t).clamp_(-1., 1.)
            mean, var = self.get_statistics(x_start, x_t, t.view(-1))
            noise = torch.randn_like(x_t)

            return mean + torch.exp(0.5 * var) * noise

    def sample(self, n, labels=None, w=3):
        self.model.eval()
        with torch.no_grad():
            x_t = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.n_steps))):
                t = (torch.ones(n, 1) * i).long().to(self.device)
                x_t = self.sample_prev(x_t, t, i, w, labels)

        x_t = (x_t.clamp(-1, 1) + 1) / 2
        x_t = (x_t * 255).type(torch.uint8)
        return x_t

    def train(self, n_epochs, data_loader, dataset_len):
        self.model.train()
        mse = nn.MSELoss()
        optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        losses = []

        for epoch in tqdm(range(n_epochs)):
            epoch_loss = 0.0
            for img, labels in tqdm(data_loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}"):
                x0 = img.to(self.device)
                y = labels.to(self.device)

                eps = torch.randn_like(x0).to(self.device)
                t = torch.randint(0, self.n_steps, (len(x0), 1)).to(self.device).long()

                if not self.is_conditioned or np.random.random() < 0.1:
                    y = None

                noisy_imgs = self.forward_process(x0, t, eps)
                predict = self.model(noisy_imgs, t, y)
                target = eps if self.type == TargetType.NOISE else x0
                loss = mse(predict, target)
                optim.zero_grad()
                loss.backward()
                optim.step()

                epoch_loss += loss.item() * len(x0) / dataset_len

            print(f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}")
            losses.append(epoch_loss)

        return losses

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def upload_weights(self, path):
        self.model.load_state_dict(torch.load(path))


