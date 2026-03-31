"""NuScenes multi-view video dataset for diffusion model training.

Supports two modes:
  - 'rgb_depth_seg': Stage 1 training with RGB images, depth maps, and segmentation masks.
  - 'render': Stage 2 training with RGB images and Gaussian-splatting render maps.

Both modes load multi-view images, road sketches, bounding boxes, captions,
and T5 caption features, with variable-resolution support.
"""

import os
import json
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_VIEWS = 6

NUS_CATEGORIES = (
    "car", "truck", "trailer", "bus", "construction_vehicle",
    "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier",
    "none",
)

CAMERA_TYPES = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT",
]


# ---------------------------------------------------------------------------
# Helper: temporal sampling
# ---------------------------------------------------------------------------

class TemporalRandomCrop:
    """Randomly crop a contiguous window of frame indices."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, total_frames: int) -> Tuple[int, int]:
        rand_end = max(0, total_frames - self.size - 1)
        begin = random.randint(0, rand_end)
        end = min(begin + self.size, total_frames)
        return begin, end


# ---------------------------------------------------------------------------
# Helpers: bounding-box manipulation
# ---------------------------------------------------------------------------

def _categories_to_caption(category_ids) -> str:
    """Build a driving-scene caption string from category indices."""
    parts = []
    for idx in category_ids:
        if idx != -1:
            parts.append(NUS_CATEGORIES[idx])
    body = ", ".join(parts) if parts else ""
    return f"Realistic driving scenes, including {body}, best quality, extremely detailed"


def _to_valid(x0, y0, x1, y1, image_size=900, min_box_size=0.005):
    """Clip box to image bounds and check validity."""
    w, h = x1 - x0, y1 - y0
    if x0 > image_size or y0 > image_size or x1 < 0 or y1 < 0:
        return False, (None, None, None, None)

    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, image_size), min(y1, image_size)

    area_ratio = (x1 - x0) * (y1 - y0) / (image_size * image_size)
    aspect = h / w if w > 0 else 0
    if area_ratio < min_box_size or aspect < 0.1 or aspect > 10:
        return False, (None, None, None, None)

    return True, (x0, y0, x1, y1)


def _recalculate_box(x0, y0, x1, y1, trans_info, crop_size, min_box_size=0.005):
    """Shift box coordinates after a crop and validate."""
    x0 -= trans_info["crop_x"]
    y0 -= trans_info["crop_y"]
    x1 -= trans_info["crop_x"]
    y1 -= trans_info["crop_y"]
    return _to_valid(x0, y0, x1, y1, crop_size, min_box_size)


def _verify_box(x0, y0, x1, y1, width, height, min_box_size):
    """Check if a box occupies enough area relative to the image."""
    return (x1 - x0) * (y1 - y0) / (width * height) >= min_box_size


# ---------------------------------------------------------------------------
# Helpers: image transforms
# ---------------------------------------------------------------------------

def _random_crop_pair(view_img, condition_img, crop_size):
    """Randomly crop two images at the same location."""
    ih, iw = view_img.shape[:2]
    assert condition_img.shape[:2] == (ih, iw)
    h1 = random.randint(0, ih - crop_size)
    w1 = random.randint(0, iw - crop_size)
    h2, w2 = h1 + crop_size, w1 + crop_size

    def _crop(img):
        return img[h1:h2, w1:w2, :] if len(img.shape) == 3 else img[h1:h2, w1:w2]

    info = {"crop_y": h1, "crop_x": w1, "perform_flip": False}
    return _crop(view_img), _crop(condition_img), info


def _crop_with_boxes(view_img, cond_img, coords_2d, category_2d, heading,
                     crop_size, min_box_size=0.005, max_trials=100):
    """Crop a single-view image pair, trying to keep bounding boxes visible."""
    assert len(coords_2d) == len(category_2d)
    assert len(coords_2d) > 1

    for trial in range(max_trials):
        out_coords, out_cat, out_head = [], [], []
        v_crop, c_crop, crop_info = _random_crop_pair(view_img, cond_img, crop_size)

        for idx, anno in enumerate(coords_2d):
            x0, y0, x1, y1 = anno
            valid, (x0, y0, x1, y1) = _recalculate_box(
                x0, y0, x1, y1, crop_info, crop_size, min_box_size)
            if valid:
                out_coords.append([x0, y0, x1, y1])
                out_cat.append(category_2d[idx])
                out_head.append(heading[idx])

        if out_cat or trial == max_trials - 1:
            return v_crop, c_crop, out_coords, out_cat, out_head

    return v_crop, c_crop, out_coords, out_cat, out_head


def _rescale_single_image(view_img, cond_img, coords_2d, category_2d,
                          heading, instance_ids, target_hw):
    """Resize a single-view image and rescale box coordinates accordingly."""
    src_h, src_w = view_img.shape[:2]
    tgt_h, tgt_w = target_hw
    sx, sy = tgt_w / src_w, tgt_h / src_h

    coords = np.array(coords_2d)
    if coords.size and coords.any():
        coords[:, 0::2] *= sx
        coords[:, 1::2] *= sy
    else:
        coords = np.array([])

    view_out = cv2.resize(view_img, (tgt_w, tgt_h))
    cond_out = cv2.resize(cond_img, (tgt_w, tgt_h))

    return (view_out, cond_out, coords,
            np.array(category_2d), np.array(heading), np.array(instance_ids))


def _create_render_mask(render_map, mask_ratio=0.05, patch_size=16):
    """Randomly mask rectangular patches in a render map (for training regularization)."""
    H, W, C = render_map.shape
    ms = min(H, W) // patch_size
    n_patches = int((H * W * mask_ratio) / (ms ** 2))

    ys = np.random.randint(0, H - ms, size=n_patches)
    xs = np.random.randint(0, W - ms, size=n_patches)

    mask = np.ones((H, W), dtype=np.uint8)
    yy, xx = np.meshgrid(np.arange(ms), np.arange(ms), indexing="ij")
    for_y = ys[:, None, None] + yy
    for_x = xs[:, None, None] + xx
    mask[for_y, for_x] = 0

    return render_map * mask[..., None]


# ---------------------------------------------------------------------------
# Utility I/O
# ---------------------------------------------------------------------------

def _load_json(path: str):
    assert os.path.exists(path), f"{path} not found"
    with open(path, "rb") as f:
        return json.loads(f.read())


def _normalize_to_tensor(img_np):
    """Normalize uint8 image to [-1, 1] float tensor [C, H, W]."""
    out = (img_np.astype(np.float32) / 127.5) - 1.0
    return torch.tensor(np.transpose(out, [2, 0, 1]))


# ---------------------------------------------------------------------------
# Main dataset
# ---------------------------------------------------------------------------

class NuScenesDataset(Dataset):
    """NuScenes multi-view video dataset for diffusion training.

    Args:
        mode: One of ``'rgb_depth_seg'`` (stage 1) or ``'render'`` (stage 2).
        data_root: Root directory for RGB images.
        road_sketch_root: Root directory for road sketch condition images.
        caption_root: Root directory for pre-extracted T5 caption features.
        embedding_root: Root directory for per-instance CLIP embeddings.
        annotation_json: Path to the JSON annotation file (sequence-level).
        depth_root: Root for depth maps (only used in ``rgb_depth_seg`` mode).
        seg_root: Root for segmentation masks (only used in ``rgb_depth_seg`` mode).
        render_root: Root for render maps (only used in ``render`` mode).
        training: Whether this is a training split.
        num_views: Number of camera views per frame (default 6).
        crop_size: Crop size for box-aware random cropping.
        min_box_size: Minimum relative box area to be considered valid.
        max_boxes_per_data: Maximum number of boxes kept per sample.
        scale_to_01: Whether to normalize box coordinates to [0, 1].
        scale_dataset: Fraction of the full dataset to use.
        frame_interval: Interval between sampled frames in a clip.
        crop_mode: Whether to apply box-aware random cropping.
        mask_render_map: Whether to randomly mask render map patches (stage 2).
        shard_index: 1-based index for dataset sharding (1 = no sharding).
        num_shards: Total number of shards (1 = no sharding).
    """

    def __init__(
        self,
        mode: str = "rgb_depth_seg",
        data_root: str = "",
        road_sketch_root: str = "",
        caption_root: str = "",
        embedding_root: str = "",
        annotation_json: str = "",
        depth_root: str = "",
        seg_root: str = "",
        render_root: str = "",
        training: bool = True,
        num_views: int = NUM_VIEWS,
        crop_size: int = 512,
        min_box_size: float = 0.02,
        max_boxes_per_data: int = 30,
        scale_to_01: bool = True,
        scale_dataset: float = 1.0,
        frame_interval: int = 1,
        crop_mode: bool = True,
        mask_render_map: bool = False,
        shard_index: int = 1,
        num_shards: int = 1,
    ):
        super().__init__()
        assert mode in ("rgb_depth_seg", "render"), f"Unknown mode: {mode}"

        self.mode = mode
        self.data_root = data_root
        self.road_sketch_root = road_sketch_root
        self.caption_root = caption_root
        self.embedding_root = embedding_root
        self.depth_root = depth_root
        self.seg_root = seg_root
        self.render_root = render_root
        self.training = training
        self.num_views = num_views
        self.crop_size = crop_size
        self.min_box_size = min_box_size
        self.max_boxes_per_data = max_boxes_per_data
        self.scale_to_01 = scale_to_01
        self.frame_interval = frame_interval
        self.crop_mode = crop_mode
        self.mask_render_map = mask_render_map

        # Load annotations
        self.data = _load_json(annotation_json)
        total = len(self.data)
        self.data = self.data[:int(total * scale_dataset)]

        # Optional sharding
        if num_shards > 1:
            idx = shard_index - 1
            interval = len(self.data) // num_shards
            if idx == num_shards - 1:
                self.data = self.data[interval * idx:]
            else:
                self.data = self.data[interval * idx: interval * (idx + 1)]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        """Index is a string ``'<idx>-<num_frames>-<height>-<width>'``."""
        try:
            return self._load_sequence(index)
        except Exception:
            return self._load_sequence(f"0-{index.split('-')[1]}-{index.split('-')[2]}-{index.split('-')[3]}"
                                       if isinstance(index, str) else index)

    # ------------------------------------------------------------------
    # Sequence loading
    # ------------------------------------------------------------------

    def _load_sequence(self, index):
        if isinstance(index, str):
            index, num_frames, height, width = [int(v) for v in index.split("-")]
        else:
            raise ValueError("Index must be a string of format 'idx-T-H-W'.")

        # Collect per-frame, per-view data
        keys = self._output_keys()
        example = {k: [] for k in keys}
        example["scene_token"] = [self.data[index][0][0]["sample_token"]]

        # Temporal sampling
        total_frames = len(self.data[index])
        temporal_sampler = TemporalRandomCrop(num_frames * self.frame_interval)
        start, end = temporal_sampler(total_frames)
        assert end - start >= num_frames, f"Not enough frames at index {index}."

        frame_indices = np.linspace(start, end - 1, num_frames, dtype=int)
        if not self.training:
            frame_indices = np.linspace(0, num_frames - 1, num_frames, dtype=int)

        for fi in frame_indices:
            for vi in range(self.num_views):
                meta = self.data[index][fi][vi]
                single = self._load_single_frame(meta, num_frames, height, width)
                for k in keys:
                    if k in single:
                        example[k].append(single[k])

        # Stack and rearrange: (F*V, ...) -> (V, F, ...)
        v = self.num_views
        example["image"] = rearrange(torch.stack(example["image"]), "(f v) ... -> v f ...", v=v)
        example["hed_edge"] = rearrange(torch.stack(example["hed_edge"]), "(f v) ... -> v f ...", v=v)
        example["mask"] = rearrange(torch.stack(example["mask"]), "(f v) ... -> v f ...", v=v)
        example["boxes_2d"] = rearrange(torch.stack(example["boxes_2d"]), "(f v) ... -> v f ...", v=v)
        example["masks"] = rearrange(torch.stack(example["masks"]), "(f v) ... -> v f ...", v=v)
        example["gt_category_2d"] = rearrange(torch.stack(example["gt_category_2d"]), "(f v) ... -> v f ...", v=v)
        example["heading"] = rearrange(torch.stack(example["heading"]), "(f v) ... -> v f ...", v=v)
        example["instance_id"] = rearrange(torch.stack(example["instance_id"]), "(f v) ... -> v f ...", v=v)
        example["positive_embedding"] = rearrange(torch.stack(example["positive_embedding"]), "(f v) ... -> v f ...", v=v)
        example["caption_feature"] = rearrange(torch.stack(example["caption_feature"]), "(f v) ... -> v f ...", v=v)
        example["attention_mask"] = rearrange(torch.cat(example["attention_mask"], dim=0), "(f v) ... -> v f ...", v=v)

        for scalar_key in ("height", "width", "ar", "num_frames", "fps"):
            example[scalar_key] = rearrange(torch.stack(example[scalar_key]), "(f v) ... -> v f ...", v=v)[:, 0]

        # Mode-specific stacking
        if self.mode == "rgb_depth_seg":
            example["depth_map"] = rearrange(torch.stack(example["depth_map"]), "(f v) ... -> v f ...", v=v)
            example["seg_map"] = rearrange(torch.stack(example["seg_map"]), "(f v) ... -> v f ...", v=v)
        else:
            example["render_map"] = rearrange(torch.stack(example["render_map"]), "(f v) ... -> v f ...", v=v)

        return example

    def _output_keys(self) -> List[str]:
        base = [
            "image", "hed_edge", "mask",
            "boxes_2d", "masks", "gt_category_2d", "heading", "instance_id",
            "positive_embedding", "caption", "caption_feature", "attention_mask",
            "height", "width", "ar", "num_frames", "fps",
            "save_name", "filename",
        ]
        if self.mode == "rgb_depth_seg":
            base += ["depth_map", "seg_map"]
        else:
            base += ["render_map"]
        return base

    # ------------------------------------------------------------------
    # Single-frame loading
    # ------------------------------------------------------------------

    def _load_single_frame(self, meta, num_frames, height, width):
        """Load and preprocess a single camera view for one frame."""
        view_path = os.path.join(self.data_root, meta["filename"])
        rs_name = "_".join(meta["road_sketch"].split("-"))
        cond_path = os.path.join(self.road_sketch_root, rs_name)

        out = {}
        out["filename"] = view_path
        out["save_name"] = f"{meta['sample_token']}-{meta['cam_type']}.png"

        # Annotations
        coords_2d = np.array(meta["gt_coords_2d"])
        category_2d = np.array(meta["gt_category_2d"])
        heading = np.array(meta["gt_heading_2d"])
        instance_ids = np.array(meta["gt_instance_id_normalized"])

        # Load RGB image
        view_img = cv2.imread(view_path)
        ori_h, ori_w = view_img.shape[:2]
        if len(view_img.shape) < 3:
            view_img = cv2.cvtColor(view_img, cv2.COLOR_GRAY2BGR)
        view_img = cv2.cvtColor(view_img, cv2.COLOR_BGR2RGB)
        view_img = np.array(view_img).astype(np.uint8)

        # Load road sketch condition
        cond_alt = cond_path.replace("_CAM", "-CAM")
        if os.path.exists(cond_path):
            cond_img = cv2.imread(cond_path)
        elif os.path.exists(cond_alt):
            cond_img = cv2.imread(cond_alt)
        else:
            cond_img = np.zeros([ori_h, ori_w, 3], dtype=np.uint8)

        if cond_img is not None and cond_img.size > 0:
            cond_img[cond_img < 125] = 0
            if len(cond_img.shape) < 3:
                cond_img = cv2.cvtColor(cond_img, cv2.COLOR_GRAY2BGR)
            cond_img = cv2.cvtColor(cond_img, cv2.COLOR_BGR2RGB)
            cond_img = np.array(cond_img).astype(np.uint8)

        assert cond_img.shape[:2] == (ori_h, ori_w)

        # Box-aware cropping
        if self.crop_mode:
            view_img, cond_img, coords_2d, category_2d, heading = _crop_with_boxes(
                view_img, cond_img, coords_2d, category_2d, heading,
                self.crop_size, self.min_box_size,
            )

        # Rescale to target resolution
        view_img, cond_img, coords_2d, category_2d, heading, instance_ids = \
            _rescale_single_image(
                view_img, cond_img, coords_2d, category_2d,
                heading, instance_ids, [height, width],
            )

        # Load mode-specific maps
        if self.mode == "rgb_depth_seg":
            depth_path = os.path.join(self.depth_root, meta["filename"].replace(".jpg", ".png"))
            depth_map = cv2.imread(depth_path)
            depth_map = cv2.resize(depth_map, (width, height))

            seg_path = os.path.join(self.seg_root, meta["filename"].replace(".jpg", ".png"))
            seg_map = cv2.imread(seg_path)
            seg_map = cv2.resize(seg_map, (width, height))

            out["depth_map"] = _normalize_to_tensor(depth_map)
            out["seg_map"] = _normalize_to_tensor(seg_map)
        else:
            render_path = os.path.join(self.render_root, meta["filename"])
            render_map = cv2.imread(render_path)
            render_map = cv2.resize(render_map, (width, height))
            if self.mask_render_map:
                render_map = _create_render_mask(render_map)
            out["render_map"] = _normalize_to_tensor(render_map)

        # Normalize images to [-1, 1]
        out["image"] = _normalize_to_tensor(view_img)
        out["hed_edge"] = _normalize_to_tensor(cond_img)
        out["mask"] = torch.tensor(1.0)

        out["boxes_2d"] = coords_2d
        out["heading"] = heading
        out["instance_id"] = instance_ids
        gt_category_2d = category_2d

        out["height"] = torch.tensor(out["image"].shape[-2])
        out["width"] = torch.tensor(out["image"].shape[-1])
        out["num_frames"] = torch.tensor(num_frames)
        out["fps"] = torch.tensor(12)
        out["ar"] = torch.tensor(out["image"].shape[-1] / out["image"].shape[-2])

        # T5 caption features
        split_dir = "train" if self.training else "val"
        token = meta["sample_token"]
        token_key = token[:-1] if len(token) == 33 else token
        attn_mask = np.load(os.path.join(self.caption_root, split_dir, "attn_mask", token_key + ".npy"))
        cap_feat = np.load(os.path.join(self.caption_root, split_dir, "caption_feature", token_key + ".npy"))
        out["attention_mask"] = torch.tensor(attn_mask)
        out["caption_feature"] = torch.tensor(cap_feat)

        # Pad / filter boxes to fixed size
        out = self._finalize_boxes(out, meta, gt_category_2d, width, height)
        return out

    # ------------------------------------------------------------------
    # Box post-processing
    # ------------------------------------------------------------------

    def _finalize_boxes(self, out, meta, gt_category_2d, width, height):
        """Pad boxes to max_boxes_per_data, load embeddings, and filter by area."""
        M = self.max_boxes_per_data
        boxes_2d = torch.zeros(M, 4)
        category = torch.full((M,), -1)
        heading = torch.zeros(M)
        instance_id = torch.zeros(M)
        masks = torch.zeros(M)
        pos_emb = torch.zeros(M, 768)

        box_arr = out["boxes_2d"]
        if not isinstance(box_arr, np.ndarray) or box_arr.size == 0 or not box_arr.any():
            out["boxes_2d"] = boxes_2d
            out["gt_category_2d"] = category
            out["heading"] = heading
            out["instance_id"] = instance_id
            out["masks"] = masks
            out["positive_embedding"] = pos_emb
            out["caption"] = _categories_to_caption(category)
            return out

        # Load per-instance CLIP embeddings
        emb_list = []
        for ins_id in meta["gt_instance_id"]:
            emb_path = os.path.join(self.embedding_root, ins_id)
            if not os.path.isfile(emb_path):
                emb_path = os.path.join(self.embedding_root, "all_densecaption", ins_id)
            emb = torch.load(emb_path, map_location="cpu")
            assert emb.shape == (1, 768)
            emb_list.append(emb.detach())
        out["positive_embedding"] = torch.cat(emb_list, dim=0)

        # Collect valid boxes sorted by area (descending)
        areas, valid_boxes, valid_cat, valid_head, valid_iid, valid_emb = [], [], [], [], [], []
        for idx, anno in enumerate(out["boxes_2d"]):
            x0, y0, x1, y1 = anno
            if not _verify_box(x0, y0, x1, y1, width, height, self.min_box_size):
                continue
            area = (x1 - x0) * (y1 - y0)
            areas.append(area)
            if self.scale_to_01:
                valid_boxes.append(torch.tensor([x0, y0, x1, y1]) / torch.tensor([width, height, width, height]))
            else:
                valid_boxes.append(torch.tensor([x0, y0, x1, y1]))
            valid_cat.append(gt_category_2d[idx])
            valid_head.append(out["heading"][idx])
            valid_iid.append(out["instance_id"][idx])
            valid_emb.append(out["positive_embedding"][idx])

        # Keep top-M by area
        if valid_boxes:
            order = torch.tensor(areas).sort(descending=True)[1][:M]
            for i, oi in enumerate(order):
                boxes_2d[i] = valid_boxes[oi]
                masks[i] = 1
                category[i] = valid_cat[oi]
                heading[i] = valid_head[oi]
                instance_id[i] = valid_iid[oi]
                pos_emb[i] = valid_emb[oi]

        out["boxes_2d"] = boxes_2d
        out["masks"] = masks
        out["gt_category_2d"] = category
        out["heading"] = heading.float()
        out["instance_id"] = instance_id.float()
        out["positive_embedding"] = pos_emb
        out["caption"] = _categories_to_caption(category)
        return out
