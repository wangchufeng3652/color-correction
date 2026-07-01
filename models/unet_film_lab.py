import torch
import torch.nn as nn
import torch.nn.functional as F



# =====================================================
# FiLM Module
# =====================================================

class FiLM(nn.Module):

    def __init__(self, vector_dim, feature_dim):

        super().__init__()

        self.gamma = nn.Linear(
            vector_dim,
            feature_dim
        )

        self.beta = nn.Linear(
            vector_dim,
            feature_dim
        )


    def forward(self, x, vec):

        gamma = self.gamma(vec)

        beta = self.beta(vec)


        gamma = gamma.unsqueeze(-1).unsqueeze(-1)

        beta = beta.unsqueeze(-1).unsqueeze(-1)


        return x * (1 + gamma) + beta




# =====================================================
# Lite UNet + FiLM + Residual (Lab)
# 与训练模型完全一致
# =====================================================

class LiteUNetFiLMResidualLab(nn.Module):

    def __init__(
            self,
            in_channels=3,
            out_channels=3,
            vector_dim=7
    ):

        super().__init__()


        def CBR(in_ch, out_ch):

            return nn.Sequential(

                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=3,
                    padding=1,
                    bias=False
                ),

                nn.BatchNorm2d(out_ch),

                nn.ReLU(inplace=True)

            )



        # -------------------------
        # Encoder
        # -------------------------

        self.enc1 = CBR(
            in_channels,
            32
        )


        self.enc2 = nn.Sequential(

            nn.MaxPool2d(2),

            CBR(
                32,
                64
            )

        )


        self.enc3 = nn.Sequential(

            nn.MaxPool2d(2),

            CBR(
                64,
                128
            )

        )



        # -------------------------
        # Bottleneck
        # -------------------------

        self.bottleneck = CBR(
            128,
            256
        )



        # -------------------------
        # Decoder
        # -------------------------

        self.up2 = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True
        )


        self.dec2 = CBR(
            256 + 128,
            128
        )


        self.up1 = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True
        )


        self.dec1 = CBR(
            128 + 64,
            64
        )



        # output

        self.out_conv = nn.Conv2d(
            64,
            out_channels,
            kernel_size=1
        )



        # -------------------------
        # FiLM layers
        # -------------------------

        self.film1 = FiLM(
            vector_dim,
            32
        )


        self.film2 = FiLM(
            vector_dim,
            64
        )


        self.film3 = FiLM(
            vector_dim,
            128
        )




    def forward(
            self,
            x_img,
            x_vec
    ):


        # Encoder

        e1 = self.enc1(x_img)

        e1 = self.film1(
            e1,
            x_vec
        )



        e2 = self.enc2(e1)

        e2 = self.film2(
            e2,
            x_vec
        )



        e3 = self.enc3(e2)

        e3 = self.film3(
            e3,
            x_vec
        )



        # bottleneck

        b = self.bottleneck(e3)



        # Decoder

        d2 = self.up2(b)



        if d2.shape[2:] != e3.shape[2:]:

            d2 = F.interpolate(
                d2,
                size=e3.shape[2:],
                mode="bilinear",
                align_corners=True
            )


        d2 = self.dec2(

            torch.cat(
                [
                    d2,
                    e3
                ],
                dim=1
            )

        )



        d1 = self.up1(d2)



        if d1.shape[2:] != e2.shape[2:]:

            d1 = F.interpolate(
                d1,
                size=e2.shape[2:],
                mode="bilinear",
                align_corners=True
            )


        d1 = self.dec1(

            torch.cat(
                [
                    d1,
                    e2
                ],
                dim=1
            )

        )



        # residual learning

        residual = self.out_conv(d1)



        residual = F.interpolate(

            residual,

            size=x_img.shape[2:],

            mode="bilinear",

            align_corners=True

        )



        out = x_img + residual



        return torch.clamp(
            out,
            0.0,
            1.0
        )