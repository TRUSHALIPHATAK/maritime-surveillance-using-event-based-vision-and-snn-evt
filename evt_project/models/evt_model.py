import torch
import torch.nn as nn


class EVTModel(nn.Module):

    def __init__(
        self,
        input_channels=2,
        image_size=128,
        patch_size=8,
        temporal_groups=10,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        mlp_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.temporal_groups = temporal_groups
        self.embed_dim = embed_dim

        self.num_patches_per_frame = (
            image_size // patch_size
        ) ** 2

        self.num_patches = (
            temporal_groups *
            self.num_patches_per_frame
        )

        # --------------------------------------------------
        # Patch embedding
        # --------------------------------------------------

        self.patch_embed = nn.Conv2d(
            input_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # --------------------------------------------------
        # Positional embeddings
        # --------------------------------------------------

        self.spatial_pos = nn.Parameter(
            torch.zeros(
                1,
                self.num_patches_per_frame,
                embed_dim
            )
        )

        self.temporal_pos = nn.Parameter(
            torch.zeros(
                1,
                temporal_groups,
                embed_dim
            )
        )

        # --------------------------------------------------
        # Transformer
        # --------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # --------------------------------------------------
        # Heatmap reconstruction
        # --------------------------------------------------

        self.heatmap_head = nn.Sequential(

            nn.Conv2d(
                embed_dim,
                64,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(64),

            nn.GELU(),

            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(32),

            nn.GELU(),

            nn.Conv2d(
                32,
                1,
                kernel_size=1,
            ),

         #nn.Sigmoid(),
        )

    def forward(self, x):

        # x:
        # [B, T, C, H, W]

        B, T, C, H, W = x.shape

        # --------------------------------------------------
        # Patch embedding
        # --------------------------------------------------

        x = x.reshape(
            B * T,
            C,
            H,
            W
        )

        x = self.patch_embed(x)

        # [B*T, embed_dim, 4, 4]

        ph = x.shape[-2]
        pw = x.shape[-1]

        x = x.flatten(2)

        # [B*T, embed_dim, patches]

        x = x.transpose(1, 2)

        # [B*T, patches, embed_dim]

        # --------------------------------------------------
        # Spatial positional encoding
        # --------------------------------------------------

        x = x + self.spatial_pos

        x = x.reshape(
            B,
            T,
            self.num_patches_per_frame,
            self.embed_dim
        )

        # --------------------------------------------------
        # Temporal positional encoding
        # --------------------------------------------------

        x = x + self.temporal_pos[:, :, None, :]

        # --------------------------------------------------
        # Flatten temporal + spatial tokens
        # --------------------------------------------------

        x = x.reshape(
            B,
            T * self.num_patches_per_frame,
            self.embed_dim
        )

        # --------------------------------------------------
        # Transformer
        # --------------------------------------------------

        x = self.transformer(x)

        # --------------------------------------------------
        # Restore temporal groups
        # --------------------------------------------------

        x = x.reshape(
            B,
            T,
            self.num_patches_per_frame,
            self.embed_dim
        )

        # --------------------------------------------------
        # Reconstruct spatial feature map
        # --------------------------------------------------

        x = x.reshape(
            B * T,
            self.num_patches_per_frame,
            self.embed_dim
        )

        x = x.transpose(1, 2)

        x = x.reshape(
            B * T,
            self.embed_dim,
            ph,
            pw
        )

        # Upsample 4x4 -> 128x128

        x = nn.functional.interpolate(
            x,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        # --------------------------------------------------
        # Heatmap
        # --------------------------------------------------

        x = self.heatmap_head(x)

        x = x.reshape(
            B,
            T,
            1,
            H,
            W
        )

        return x


if __name__ == "__main__":

    model = EVTModel()

    x = torch.randn(
        2,
        10,
        2,
        128,
        128
    )

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Total parameters:", total)
    print("Trainable parameters:", trainable)