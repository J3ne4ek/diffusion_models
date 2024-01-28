import torch
import torch.nn as nn
import torch.nn.functional as F


class Resnet(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.enc1 = nn.Sequential(nn.GroupNorm(1, in_channels), nn.SiLU(),
                                  nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))

        self.enc2 = nn.Sequential(nn.GroupNorm(1, out_channels), nn.SiLU(),
                                  nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

        if in_channels != out_channels:
            self.sample = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.sample = None

        self.t = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

    def forward(self, x, t):
        h = self.enc1(x) + self.t(t).reshape(len(x), -1, 1, 1)
        h = self.enc2(h)

        if self.sample:
            x = self.sample(x)

        return h + x


class SelfAttention(nn.Module):
    def __init__(self, channels, img_size):
        super().__init__()
        self.channels = channels
        self.size = img_size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.mlp = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value += x
        attention_value = self.mlp(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class Encode(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, time_emb_dim, with_attention=False):
        super().__init__()
        self.enc = Resnet(in_channels, out_channels, time_emb_dim)
        self.att = SelfAttention(out_channels, img_size) if with_attention else None
        self.enc2 = Resnet(out_channels, out_channels, time_emb_dim)
        self.att2 = SelfAttention(out_channels, img_size) if with_attention else None
        self.pool = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, t):
        x = self.enc(x, t)
        x = self.enc2(self.att(x), t) if self.att is not None else self.enc2(x, t)
        x = self.pool(self.att2(x)) if self.att2 is not None else self.pool(x)

        return x


class Decode(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, time_emb_dim, with_attention=False):
        super().__init__()
        self.unpool = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.dec = Resnet(in_channels, out_channels, time_emb_dim)
        self.att = SelfAttention(out_channels, img_size) if with_attention else None
        self.dec2 = Resnet(out_channels, out_channels, time_emb_dim)
        self.att2 = SelfAttention(out_channels, img_size) if with_attention else None

    def forward(self, x, e, t):
        x = self.unpool(x)
        x = self.dec(torch.cat([x, e], dim=1), t)
        x = self.dec2(self.att(x), t) if self.att is not None else self.dec2(x, t)
        x = self.att2(x) if self.att2 else x
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_steps=1000, dim=16, img_size=16, n_classes=None):
        super().__init__()

        self.time_embed = nn.Embedding(n_steps, dim)
        self.time_embed.weight.data = self.sinusoidal_embedding(n_steps, dim)
        self.time_embed.requires_grad_(False)
        time_emb_dim = dim * 4

        self.t = nn.Sequential(nn.Linear(dim, time_emb_dim),
                               nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim))

        if n_classes:
            self.class_embed = nn.Embedding(n_classes, dim)
            self.c = nn.Sequential(nn.Linear(dim, time_emb_dim),
                                   nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim))

        self.enc1 = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)

        self.enc2 = Encode(dim, dim * 2, img_size, time_emb_dim)
        self.enc3 = Encode(dim * 2, dim * 4, img_size // 2, time_emb_dim, with_attention=True)
        self.enc4 = Encode(dim * 4, dim * 8, img_size // 4, time_emb_dim, with_attention=True)

        self.b1 = Resnet(dim * 8, dim * 16, time_emb_dim)
        self.att1 = SelfAttention(dim * 16, img_size // 8)
        self.b2 = Resnet(dim * 16, dim * 16, time_emb_dim)
        self.att2 = SelfAttention(dim * 16, img_size // 8)
        self.b3 = Resnet(dim * 16, dim * 8, time_emb_dim)

        self.dec1 = Decode(dim * 8, dim * 4, img_size // 4, time_emb_dim, with_attention=True)
        self.dec2 = Decode(dim * 4, dim * 2, img_size // 2, time_emb_dim, with_attention=True)
        self.dec3 = Decode(dim * 2, dim, img_size, time_emb_dim)

        self.outconv = nn.Sequential(nn.GroupNorm(1, dim), nn.SiLU(),
                                     nn.Conv2d(dim, out_channels, kernel_size=1))

    def sinusoidal_embedding(self, n, d):
        embedding = torch.zeros(n, d)
        wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)]).reshape((1, d))
        t = torch.arange(n).reshape((n, 1))
        embedding[:, ::2] = torch.sin(t * wk[:, ::2])
        embedding[:, 1::2] = torch.cos(t * wk[:, ::2])

        return embedding

    def forward(self, x, t, y=None):
        t = self.time_embed(t)
        t = self.t(t)
        if y is not None:
            c = self.class_embed(y)
            c = self.c(c)
            t += c

        e1 = self.enc1(x)

        e2 = self.enc2(e1, t)
        e3 = self.enc3(e2, t)
        e4 = self.enc4(e3, t)

        e5 = self.b1(e4, t)
        e5 = self.b2(self.att1(e5), t)
        e5 = self.b3(self.att2(e5), t)

        d1 = self.dec1(e5, e3, t)
        d2 = self.dec2(d1, e2, t)
        d3 = self.dec3(d2, e1, t)

        out = self.outconv(d3)

        return out
