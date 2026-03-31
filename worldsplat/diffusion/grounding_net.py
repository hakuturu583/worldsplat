"""Grounding network that fuses text embeddings with spatial box, heading, and instance ID information."""

import torch
import torch.nn as nn


class FourierEmbedder:
    """Encodes continuous values into Fourier frequency features (sin/cos pairs)."""

    def __init__(self, num_freqs: int = 64, temperature: float = 100.0):
        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** (torch.arange(num_freqs) / num_freqs)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, cat_dim: int = -1) -> torch.Tensor:
        out = []
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, cat_dim)


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out all parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class PositionNet(nn.Module):
    """Grounding network that produces per-object token embeddings.

    Fuses text/positive embeddings with Fourier-encoded 2D bounding boxes,
    heading angles, and instance IDs. Unused object slots are replaced with
    learnable null embeddings based on the provided masks.

    Args:
        in_dim: Dimension of the positive (text) embeddings.
        out_dim: Output token embedding dimension.
        heading_dim: Unused legacy parameter kept for checkpoint compatibility.
        fourier_freqs: Number of Fourier frequency bands for positional encoding.
    """

    def __init__(self, in_dim: int, out_dim: int, heading_dim: int = 32, fourier_freqs: int = 8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Fourier embedding for 2D boxes (4 coords: x1, y1, x2, y2)
        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs * 2 * 4  # sin+cos for each of 4 box coords

        # Main fusion MLP: text embedding + position embedding -> output tokens
        self.linears = nn.Sequential(
            nn.Linear(self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        # Learnable null embeddings for masked (padding) slots
        self.null_positive_feature = nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = nn.Parameter(torch.zeros([self.position_dim]))

        # Heading encoder: Fourier embedding + zero-initialized MLP
        self.heading_fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.heading_dim = fourier_freqs * 2 * 1  # sin+cos for 1 scalar

        self.heading_mlp = nn.Sequential(
            nn.Linear(self.heading_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            zero_module(nn.Linear(32, self.position_dim)),
        )

        # Instance ID encoder: Fourier embedding + zero-initialized MLP
        self.instance_id_mlp = nn.Sequential(
            nn.Linear(self.heading_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            zero_module(nn.Linear(32, self.position_dim)),
        )

        self.null_heading_feature = nn.Parameter(torch.zeros([self.heading_dim]))
        self.null_instance_id_feature = nn.Parameter(torch.zeros([self.heading_dim]))

    def forward(self, grounding_input: dict) -> torch.Tensor:
        """Produce per-object grounding token embeddings.

        Args:
            grounding_input: Dict with keys:
                - boxes_2d: [B, N, 4] normalized 2D bounding boxes
                - masks: [B, N] binary mask (1 = valid object, 0 = padding)
                - heading: [B, N] heading angles in degrees (-180 to 180)
                - instance_id: [B, N] instance ID scalars
                - positive_embeddings: [B, N, D] text/positive embeddings

        Returns:
            Object token embeddings of shape [B, N, out_dim].
        """
        boxes_2d = grounding_input["boxes_2d"]
        masks = grounding_input["masks"]
        heading = grounding_input["heading"]
        positive_embeddings = grounding_input["positive_embeddings"]
        instance_id = grounding_input["instance_id"]

        # Normalize heading from [-180, 180] to [0, 1]
        heading = (heading + 180) / 360

        B, N, _ = boxes_2d.shape
        masks = masks.unsqueeze(-1)
        heading = heading.unsqueeze(-1)
        instance_id = instance_id.unsqueeze(-1)

        # Encode heading with Fourier features + zero-init MLP
        heading_emb = self.heading_fourier_embedder(heading)  # [B, N, heading_dim]
        heading_null = self.null_heading_feature.view(1, 1, -1)
        heading_emb = heading_emb * masks + (1 - masks) * heading_null
        heading_offset = self.heading_mlp(heading_emb)  # [B, N, position_dim]

        # Encode instance ID with Fourier features + zero-init MLP
        instance_id_emb = self.heading_fourier_embedder(instance_id)  # [B, N, heading_dim]
        instance_id_null = self.null_instance_id_feature.view(1, 1, -1)
        instance_id_emb = instance_id_emb * masks + (1 - masks) * instance_id_null
        instance_id_offset = self.instance_id_mlp(instance_id_emb)  # [B, N, position_dim]

        # Encode 2D box coordinates with Fourier features, add heading and ID offsets
        box_emb = self.fourier_embedder(boxes_2d)  # [B, N, position_dim]
        box_emb = box_emb + heading_offset + instance_id_offset

        # Replace padding slots with learnable null embeddings
        positive_null = self.null_positive_feature.view(1, 1, -1)
        box_null = self.null_position_feature.view(1, 1, -1)
        positive_embeddings = positive_embeddings * masks + (1 - masks) * positive_null
        box_emb = box_emb * masks + (1 - masks) * box_null

        # Fuse text and spatial features
        objs = self.linears(torch.cat([positive_embeddings, box_emb], dim=-1))
        assert objs.shape == torch.Size([B, N, self.out_dim])
        return objs
