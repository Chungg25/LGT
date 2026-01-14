import torch
from torch import nn
import torch.nn.functional as F


class PatchChannelGLU(nn.Module):
    def __init__(self, patch_len, d_model):
        super().__init__()
        self.linear_a = nn.Linear(patch_len, d_model)
        self.linear_b = nn.Linear(patch_len, d_model)

    def forward(self, x):  # x: [B*C, patch_num, patch_len]
        a = self.linear_a(x)
        b = torch.sigmoid(self.linear_b(x))
        return a * b


class LocalTemporal(nn.Module):
    def __init__(self, kernel_size, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            dilation=dilation,
            padding= (kernel_size - 1) // 2 * dilation,
            groups=1
        )

    def forward(self, x):

        return self.conv(x)


class AdaptiveFusion(nn.Module):
    def __init__(self, pred_len):
        super().__init__()
        self.fc = nn.Linear(pred_len, pred_len)

    def forward(self, s, t):
        alpha = torch.sigmoid(self.fc(s + t))
        return alpha * s + (1 - alpha) * t


class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,
                 dropout=0.1, d_model=64, nhead=4, num_layers=2):
        super().__init__()

        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch

        patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        self.patch_num = patch_num 

        self.patch_conv = LocalTemporal(kernel_size=3, dilation=1)

        self.patch_glu = PatchChannelGLU(patch_len, d_model)

        self.patch_embed = nn.Linear(d_model, d_model)
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 2,
                # dim_feedforward = 1024,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            ),
            num_layers=num_layers
        )


        self.flatten = nn.Flatten(start_dim=-2)
        self.linear_seasonal = nn.Linear(self.patch_num * d_model, pred_len * 2)
        self.gelu_seasonal = nn.GELU()
        self.dropout_seasonal = nn.Dropout(dropout)
        self.linear_seasonal2 = nn.Linear(pred_len * 2, pred_len)


        self.linear_trend = nn.Linear(seq_len, pred_len * 2)
        self.avg_trend = nn.AvgPool1d(kernel_size=2)
        self.ln_trend = nn.LayerNorm(pred_len)
        self.gelu_trend = nn.GELU()
        self.dropout_trend = nn.Dropout(dropout)
        self.linear_trend2 = nn.Linear(pred_len, pred_len)
        
        self.adaptive_fusion = AdaptiveFusion(pred_len)

    def forward(self, s, t):
        # s, t: [B, seq_len, C]
        s = s.permute(0, 2, 1)  # [B, C, seq_len]
        t = t.permute(0, 2, 1)
        B, C, I = s.shape

        s_flat = s.reshape(B * C, I)
        if self.padding_patch == 'end':
            s_flat = self.padding_patch_layer(s_flat)

        s_patch = s_flat.unfold(
            dimension=-1,
            size=self.patch_len,
            step=self.stride
        )  # [B*C, patch_num, patch_len]

        BC, P, L = s_patch.shape
        s_patch = s_patch.reshape(BC * P, 1, L)
        residual = s_patch
        s_patch = self.patch_conv(s_patch)
        s_patch = s_patch + residual
        s_patch = s_patch.reshape(BC, P, L)

        s_patch = self.patch_glu(s_patch)      
        s_patch = F.gelu(s_patch)
        s_patch = self.patch_embed(s_patch)

        s_patch_residual = s_patch
        s_patch = self.transformer_encoder(s_patch)
        s_patch = s_patch + s_patch_residual

        s_patch = self.flatten(s_patch)
        s = self.linear_seasonal(s_patch)
        s = self.gelu_seasonal(s)
        s = self.dropout_seasonal(s)
        s = self.linear_seasonal2(s)

        t = t.reshape(B * C, I)

        t = self.linear_trend(t)
        t = self.avg_trend(t)
        t = self.ln_trend(t)
        t = self.gelu_trend(t)
        t = self.dropout_trend(t)
        t = self.linear_trend2(t)


        x = self.adaptive_fusion(s, t)
        x = x.view(B, C, self.pred_len)
        x = x.permute(0, 2, 1)                
        return x
