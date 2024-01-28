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
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = None

        self.t = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

    def forward(self, x, t):
        h = self.enc1(x) + self.t(t).reshape(len(x), -1, 1, 1)
        h = self.enc2(h)

        if self.downsample:
            x = self.downsample(x)

        return h + x


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x, t):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_steps=1000, dim=16, n_classes=None):
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

        self.enc1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)

        self.enc2 = Resnet(16, 32, time_emb_dim)
        self.enc22 = Resnet(32, 32, time_emb_dim)
        self.pool2 = nn.Conv2d(32, 32, kernel_size=2, stride=2)

        self.enc3 = Resnet(32, 64, time_emb_dim)
        self.att1 = SelfAttention(64, 8)
        self.enc32 = Resnet(64, 64, time_emb_dim)
        self.att2 = SelfAttention(64, 8)
        self.pool3 = nn.Conv2d(64, 64, kernel_size=2, stride=2)

        self.enc4 = Resnet(64, 128, time_emb_dim)
        self.att3 = SelfAttention(128, 4)
        self.enc42 = Resnet(128, 128, time_emb_dim)
        self.att4 = SelfAttention(128, 4)
        self.pool4 = nn.Conv2d(128, 128, kernel_size=2, stride=2)

        self.b1 = Resnet(128, 256, time_emb_dim)
        self.b2 = SelfAttention(256, 2)
        self.b3 = Resnet(256, 256, time_emb_dim)
        self.b4 = SelfAttention(256, 2)
        self.b5 = Resnet(256, 128, time_emb_dim)

        self.unpool1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = Resnet(128, 64, time_emb_dim)
        self.att5 = SelfAttention(64, 4)
        self.dec12 = Resnet(64, 64, time_emb_dim)
        self.att6 = SelfAttention(64, 4)

        self.unpool2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = Resnet(64, 32, time_emb_dim)
        self.att7 = SelfAttention(32, 8)
        self.dec22 = Resnet(32, 32, time_emb_dim)
        self.att8 = SelfAttention(32, 8)

        self.unpool3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec3 = Resnet(32, 16, time_emb_dim)
        self.dec32 = Resnet(16, 16, time_emb_dim)

        self.outconv = nn.Sequential(nn.GroupNorm(1, 16), nn.SiLU(),
                                     nn.Conv2d(16, out_channels, kernel_size=1))

    def sinusoidal_embedding(self, n, d):
        embedding = torch.zeros(n, d)
        wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
        wk = wk.reshape((1, d))
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
        e2 = self.enc22(e2, t)
        e2 = self.pool2(e2)

        e3 = self.enc3(e2, t)
        e3 = self.enc32(self.att1(e3, t), t)
        e3 = self.pool3(self.att2(e3, t))

        e4 = self.enc4(e3, t)
        e4 = self.enc42(self.att3(e4, t), t)
        e4 = self.pool4(self.att4(e4, t))

        e5 = self.b1(e4, t)
        e5 = self.b2(e5, t)
        e5 = self.b3(e5, t)
        e5 = self.b4(e5, t)
        e5 = self.b5(e5, t)

        d1 = self.unpool1(e5)
        d1 = self.dec1(torch.cat([d1, e3], dim=1), t)
        d1 = self.dec12(self.att5(d1, t), t)

        d2 = self.unpool2(self.att6(d1, t))
        d2 = self.dec2(torch.cat([d2, e2], dim=1), t)
        d2 = self.dec22(self.att7(d2, t), t)

        d3 = self.unpool3(self.att8(d2, t))
        d3 = self.dec3(torch.cat([d3, e1], dim=1), t)
        d3 = self.dec32(d3, t)

        out = self.outconv(d3)

        return out
