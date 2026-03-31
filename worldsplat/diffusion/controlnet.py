"""
ControlNet wrapper for STDiT2.

Unified controller supporting both stage 1 (multi-channel fused input) and
stage 2 (standard single-channel input). The behaviour is selected
automatically based on ``base_model.multi_channels``.
"""

import torch
import torch.nn as nn
from copy import deepcopy
from einops import rearrange

from .stdit2 import STDiT2, STDiT2Block, auto_grad_checkpoint


# =====================================================================
# ControlNet Block (copied transformer block with projections)
# =====================================================================


class ControlDitBlock(nn.Module):
    """A single ControlNet block: a deep copy of a base STDiT2Block plus
    learnable input/output projection layers (zero-initialized)."""

    def __init__(self, base_block: STDiT2Block, block_index: int = 0):
        super().__init__()
        self.copied_block = deepcopy(base_block)
        self.block_index = block_index

        for p in self.copied_block.parameters():
            p.requires_grad_(True)
        self.copied_block.load_state_dict(base_block.state_dict())
        self.copied_block.train()

        hidden_size = base_block.hidden_size
        self.hidden_size = hidden_size

        # Zero-initialized projections
        if self.block_index == 0:
            self.before_proj = nn.Linear(hidden_size, hidden_size)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, x, y, t, t_tmp, mask=None, x_mask=None, t0=None, t0_tmp=None, T=None, S=None, c=None):
        if self.block_index == 0:
            c = self.before_proj(c)
            c = self.copied_block(x + c, y, t, t_tmp, mask, x_mask, t0, t0_tmp, T, S, inflated=False)
        else:
            c = self.copied_block(c, y, t, t_tmp, mask, x_mask, t0, t0_tmp, T, S, inflated=False)
        c_skip = self.after_proj(c)
        return c, c_skip


# =====================================================================
# ControlNet Wrapper
# =====================================================================


class ControlSTDiT2(nn.Module):
    """ControlNet wrapper around an STDiT2 base model.

    When ``base_model.multi_channels > 1`` (stage 1), the main noisy input is
    embedded via ``x_embedder_fuse`` (multi-channel), while the control signal
    ``c`` uses ``x_embedder`` (single-channel). When ``multi_channels == 1``
    (stage 2), both use ``x_embedder``.

    Args:
        base_model: A pre-built :class:`STDiT2` instance.
        copy_blocks_num: Number of base-model blocks to duplicate for the
            control branch (default 13 out of 28).
        position_net: An ``nn.Module`` that encodes grounding box features
            (e.g. Fourier embeddings). Its output is projected to the model's
            hidden dimension before being injected as cross-attention tokens.
    """

    def __init__(
        self,
        base_model: STDiT2,
        copy_blocks_num: int = 13,
        position_net: nn.Module = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.copy_blocks_num = copy_blocks_num
        self.total_blocks_num = len(base_model.blocks)

        for p in self.base_model.parameters():
            p.requires_grad_(True)

        # Build ControlNet branch (deep copies of the first N blocks)
        self.controlnet = nn.ModuleList(
            [ControlDitBlock(base_model.blocks[i], i) for i in range(copy_blocks_num)]
        )

        # Grounding / box conditioning
        self.position_net = position_net
        self.box_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(768, self.base_model.hidden_size),
        )

    # Delegate attribute access to the base model for convenience
    def __getattr__(self, name: str):
        if name in ("forward",):
            return self.__dict__[name]
        elif name in ("base_model", "controlnet", "position_net", "box_proj"):
            return super().__getattr__(name)
        else:
            return getattr(self.base_model, name)

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        x,
        timestep,
        y,
        mask=None,
        x_mask=None,
        num_frames=None,
        height=None,
        width=None,
        ar=None,
        fps=None,
        data_info=None,
        c=None,
        box_mask=None,
    ):
        """
        Args:
            x: Noisy latent ``[B, C, T, H, W]``.  For stage 1 this has
                ``C = in_channels * multi_channels``.
            timestep: Diffusion steps ``[B]`` or ``[B, T]`` (FIFO mode).
            y: Caption embeddings ``[B, 1, N_token, C]``.
            c: Control signal (rendered latent) ``[B, C_ctrl, T, H, W]`` or
                ``None`` for unconditional pass.
            data_info: Dict containing ``'box'`` tensor ``[B*T, N_box, C_box]``.
            box_mask: Bool mask ``[B*T, N_box]`` for valid boxes.
            (Other args forwarded from the base model.)

        Returns:
            Output latent ``[B, C_out, T, H, W]``.
        """
        is_fifo = timestep.ndim == 2
        assert not (is_fifo and x_mask is not None)

        # --- Box grounding ---
        box = data_info["box"]
        box = self.box_proj(self.position_net(box))
        if x.shape[0] * x.shape[2] != box.shape[0]:
            box = torch.cat([box] * 2)
            box_mask = torch.cat([box_mask] * 2)

        B = x.shape[0]
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)

        # === Conditioning embeddings ===
        hw = torch.cat([height[:, None], width[:, None]], dim=1)
        rs = (height[0].item() * width[0].item()) ** 0.5
        csize = self.csize_embedder(hw, B)

        ar = ar.unsqueeze(1)
        ar = self.ar_embedder(ar, B)
        cond_info = torch.cat([csize, ar], dim=1)

        fl = num_frames.unsqueeze(1)
        fps = fps.unsqueeze(1)
        fl = self.fl_embedder(fl, B)
        fl = fl + self.fps_embedder(fps, B)

        # === Spatial layout ===
        _, _, Tx, Hx, Wx = x.size()
        T, H, W = self.base_model.get_dynamic_size(x)
        S = H * W
        scale = rs / self.input_sq_size
        base_size = round(S**0.5)
        pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size)

        # === Patch embedding (stage-dependent) ===
        if self.base_model.multi_channels > 1:
            x = self.x_embedder_fuse(x)
        else:
            x = self.x_embedder(x)
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        x = x + pos_emb
        x = rearrange(x, "B T S C -> B (T S) C")

        # Embed control signal (always single-channel embedder)
        if c is not None:
            if c.shape[0] != x.shape[0]:
                c = torch.cat([c] * 2)
            c = c.to(self.dtype)
            c_pos_emb = self.pos_embed(c, H, W, scale=scale, base_size=base_size)
            c = self.x_embedder(c)
            c = rearrange(c, "B (T S) C -> B T S C", T=T, S=S)
            c = c + c_pos_emb
            c = rearrange(c, "B T S C -> B (T S) C")

        # === Timestep embedding (with FIFO support) ===
        if is_fifo:
            timestep = rearrange(timestep, "B T -> (B T)")
            t = self.t_embedder(timestep, dtype=x.dtype)
            t = rearrange(t, "(B T) C -> B T C", T=T)
            t_spc = t + cond_info.unsqueeze(1)
            t_tmp = t + fl.unsqueeze(1)
            t_spc_mlp = self.t_block(t_spc)
            t_tmp_mlp = self.t_block_temp(t_tmp)
            if x_mask is not None:
                t0_timestep = torch.zeros_like(timestep)
                t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
                t0 = rearrange(t0, "(B T) C -> B T C", T=T)
                t0_spc = t0 + cond_info.unsqueeze(1)
                t0_tmp = t0 + fl.unsqueeze(1)
                t0_spc_mlp = self.t_block(t0_spc)
                t0_tmp_mlp = self.t_block_temp(t0_tmp)
            else:
                t0_spc = None
                t0_tmp = None
                t0_spc_mlp = None
                t0_tmp_mlp = None
            timestep = rearrange(timestep, "(B T) -> B T", T=T)
        else:
            t = self.t_embedder(timestep, dtype=x.dtype)
            t_spc = t + cond_info
            t_tmp = t + fl
            t_spc_mlp = self.t_block(t_spc)
            t_tmp_mlp = self.t_block_temp(t_tmp)
            if x_mask is not None:
                t0_timestep = torch.zeros_like(timestep)
                t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
                t0_spc = t0 + cond_info
                t0_tmp = t0 + fl
                t0_spc_mlp = self.t_block(t0_spc)
                t0_tmp_mlp = self.t_block_temp(t0_tmp)
            else:
                t0_spc = None
                t0_tmp = None
                t0_spc_mlp = None
                t0_tmp_mlp = None

        # === Caption embedding + box grounding tokens ===
        y = self.y_embedder(y, self.training)
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            if box_mask is not None:
                mask = torch.cat([mask, box_mask], dim=-1)
                y = torch.cat([y, box.unsqueeze(1)], dim=-2)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        # === Block args (shared by base and control blocks) ===
        block_args = (y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T, S)

        # === Run first base block ===
        x = auto_grad_checkpoint(self.base_model.blocks[0], x, *block_args)

        # === ControlNet branch ===
        if c is not None:
            for index in range(1, self.copy_blocks_num + 1):
                c, c_skip = auto_grad_checkpoint(
                    self.controlnet[index - 1], x, *block_args, c,
                )
                x = auto_grad_checkpoint(
                    self.base_model.blocks[index], x + c_skip, *block_args,
                )
            for index in range(self.copy_blocks_num + 1, self.total_blocks_num):
                x = auto_grad_checkpoint(self.base_model.blocks[index], x, *block_args)
        else:
            for index in range(1, self.total_blocks_num):
                x = auto_grad_checkpoint(self.base_model.blocks[index], x, *block_args)

        # === Final layer + unpatchify ===
        x = self.final_layer(x, t, x_mask, t0_spc, T, S)
        x = self.base_model.unpatchify(x, T, H, W, Tx, Hx, Wx)
        x = x.to(torch.float32)
        return x
