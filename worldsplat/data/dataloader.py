"""Bucket-based variable-resolution dataloader for multi-view video training.

Includes:
  - Aspect ratio tables and resolution buckets (from OpenSora).
  - ``Bucket`` class for assigning samples to resolution/frame-count bins.
  - ``VariableVideoBatchSampler`` for distributed, bucket-aware batching.
  - ``prepare_dataloader`` factory function.
"""

import logging
import math
import random
import warnings
from collections import OrderedDict, defaultdict
from pprint import pformat
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)


# =====================================================================
# Aspect-ratio tables
# =====================================================================

def _compute_hw(aspect: float, total_pixels: int, eps: float = 1e-4) -> Tuple[int, int]:
    """Compute even (H, W) for a given aspect ratio and pixel budget."""
    h = (total_pixels * aspect) ** 0.5 + eps
    h = math.ceil(h) if math.ceil(h) % 2 == 0 else math.floor(h)
    w = h / aspect + eps
    w = math.ceil(w) if math.ceil(w) % 2 == 0 else math.floor(w)
    return h, w


ASPECT_RATIO_144P = {
    "0.38": (117, 312), "0.43": (125, 291), "0.48": (133, 277),
    "0.50": (135, 270), "0.53": (139, 262), "0.54": (141, 260),
    "0.56": (144, 256), "0.62": (151, 241), "0.67": (156, 234),
    "0.75": (166, 221), "1.00": (192, 192), "1.33": (221, 165),
    "1.50": (235, 156), "1.78": (256, 144), "1.89": (263, 139),
    "2.00": (271, 135), "2.08": (277, 132),
}

ASPECT_RATIO_256 = {
    "0.25": (128, 512), "0.26": (128, 496), "0.27": (128, 480),
    "0.28": (128, 464), "0.32": (144, 448), "0.33": (144, 432),
    "0.35": (144, 416), "0.4": (160, 400), "0.42": (160, 384),
    "0.48": (176, 368), "0.5": (176, 352), "0.52": (176, 336),
    "0.57": (192, 336), "0.6": (192, 320), "0.68": (208, 304),
    "0.72": (208, 288), "0.78": (224, 288), "0.82": (224, 272),
    "0.88": (240, 272), "0.94": (240, 256), "1.0": (256, 256),
    "1.07": (256, 240), "1.13": (272, 240), "1.21": (272, 224),
    "1.29": (288, 224), "1.38": (288, 208), "1.46": (304, 208),
    "1.67": (320, 192), "1.75": (336, 192), "2.0": (352, 176),
    "2.09": (368, 176), "2.4": (384, 160), "2.5": (400, 160),
    "2.89": (416, 144), "3.0": (432, 144), "3.11": (448, 144),
    "3.62": (464, 128), "3.75": (480, 128), "3.88": (496, 128),
    "4.0": (512, 128),
}

ASPECT_RATIO_240P = {
    "0.38": (196, 522), "0.43": (210, 490), "0.48": (222, 462),
    "0.50": (226, 452), "0.53": (232, 438), "0.54": (236, 436),
    "0.56": (240, 426), "0.62": (252, 404), "0.67": (262, 393),
    "0.75": (276, 368), "1.00": (320, 320), "1.33": (370, 278),
    "1.50": (392, 262), "1.78": (426, 240), "1.89": (440, 232),
    "2.00": (452, 226), "2.08": (462, 222),
}

ASPECT_RATIO_360P = {
    "0.38": (294, 784), "0.43": (314, 732), "0.48": (332, 692),
    "0.50": (340, 680), "0.53": (350, 662), "0.54": (352, 652),
    "0.56": (360, 640), "0.62": (380, 608), "0.67": (392, 588),
    "0.75": (416, 554), "1.00": (480, 480), "1.33": (554, 416),
    "1.50": (588, 392), "1.78": (640, 360), "1.89": (660, 350),
    "2.00": (678, 340), "2.08": (692, 332),
}

ASPECT_RATIO_512 = {
    "0.25": (256, 1024), "0.26": (256, 992), "0.27": (256, 960),
    "0.28": (256, 928), "0.32": (288, 896), "0.33": (288, 864),
    "0.35": (288, 832), "0.4": (320, 800), "0.42": (320, 768),
    "0.48": (352, 736), "0.5": (352, 704), "0.52": (352, 672),
    "0.57": (384, 672), "0.6": (384, 640), "0.68": (416, 608),
    "0.72": (416, 576), "0.78": (448, 576), "0.82": (448, 544),
    "0.88": (480, 544), "0.94": (480, 512), "1.0": (512, 512),
    "1.07": (512, 480), "1.13": (544, 480), "1.21": (544, 448),
    "1.29": (576, 448), "1.38": (576, 416), "1.46": (608, 416),
    "1.67": (640, 384), "1.75": (672, 384), "2.0": (704, 352),
    "2.09": (736, 352), "2.4": (768, 320), "2.5": (800, 320),
    "2.89": (832, 288), "3.0": (864, 288), "3.11": (896, 288),
    "3.62": (928, 256), "3.75": (960, 256), "3.88": (992, 256),
    "4.0": (1024, 256),
}

ASPECT_RATIO_480P = {
    "0.38": (392, 1046), "0.43": (420, 980), "0.48": (444, 925),
    "0.50": (452, 904), "0.53": (466, 880), "0.54": (470, 870),
    "0.56": (480, 854), "0.62": (506, 810), "0.67": (522, 784),
    "0.75": (554, 738), "1.00": (640, 640), "1.33": (740, 555),
    "1.50": (784, 522), "1.78": (854, 480), "1.89": (880, 466),
    "2.00": (906, 454), "2.08": (924, 444),
}

ASPECT_RATIO_720P = {
    "0.38": (588, 1568), "0.43": (628, 1466), "0.48": (666, 1388),
    "0.50": (678, 1356), "0.53": (698, 1318), "0.54": (706, 1306),
    "0.56": (720, 1280), "0.62": (758, 1212), "0.67": (784, 1176),
    "0.75": (832, 1110), "1.00": (960, 960), "1.33": (1108, 832),
    "1.50": (1176, 784), "1.78": (1280, 720), "1.89": (1320, 698),
    "2.00": (1358, 680), "2.08": (1386, 666),
}

ASPECT_RATIO_900P = {
    "0.38": (738, 1946), "0.43": (878, 1830), "0.48": (830, 1732),
    "0.50": (848, 1696), "0.53": (874, 1648), "0.54": (882, 1632),
    "0.56": (900, 1600), "0.62": (758, 1212), "0.67": (784, 1176),
    "0.75": (832, 1110), "1.00": (1200, 1200), "1.33": (1040, 782),
    "1.50": (980, 652), "1.78": (900, 506), "1.89": (872, 462),
    "2.00": (848, 424), "2.08": (832, 400),
}

ASPECT_RATIO_1024 = {
    "0.25": (512, 2048), "0.26": (512, 1984), "0.27": (512, 1920),
    "0.28": (512, 1856), "0.32": (576, 1792), "0.33": (576, 1728),
    "0.35": (576, 1664), "0.4": (640, 1600), "0.42": (640, 1536),
    "0.48": (704, 1472), "0.5": (704, 1408), "0.52": (704, 1344),
    "0.57": (768, 1344), "0.6": (768, 1280), "0.68": (832, 1216),
    "0.72": (832, 1152), "0.78": (896, 1152), "0.82": (896, 1088),
    "0.88": (960, 1088), "0.94": (960, 1024), "1.0": (1024, 1024),
    "1.07": (1024, 960), "1.13": (1088, 960), "1.21": (1088, 896),
    "1.29": (1152, 896), "1.38": (1152, 832), "1.46": (1216, 832),
    "1.67": (1280, 768), "1.75": (1344, 768), "2.0": (1408, 704),
    "2.09": (1472, 704), "2.4": (1536, 640), "2.5": (1600, 640),
    "2.89": (1664, 576), "3.0": (1728, 576), "3.11": (1792, 576),
    "3.62": (1856, 512), "3.75": (1920, 512), "3.88": (1984, 512),
    "4.0": (2048, 512),
}

ASPECT_RATIO_1080P = {
    "0.38": (882, 2352), "0.43": (942, 2198), "0.48": (998, 2080),
    "0.50": (1018, 2036), "0.53": (1048, 1980), "0.54": (1058, 1958),
    "0.56": (1080, 1920), "0.62": (1138, 1820), "0.67": (1176, 1764),
    "0.75": (1248, 1664), "1.00": (1440, 1440), "1.33": (1662, 1246),
    "1.50": (1764, 1176), "1.78": (1920, 1080), "1.89": (1980, 1048),
    "2.00": (2036, 1018), "2.08": (2078, 998),
}

# Registry: name -> (num_pixels, aspect_ratio_dict)
ASPECT_RATIOS: Dict[str, Tuple[int, dict]] = {
    "144p":  (36864,   ASPECT_RATIO_144P),
    "256":   (65536,   ASPECT_RATIO_256),
    "240p":  (102240,  ASPECT_RATIO_240P),
    "360p":  (230400,  ASPECT_RATIO_360P),
    "512":   (262144,  ASPECT_RATIO_512),
    "480p":  (409920,  ASPECT_RATIO_480P),
    "720p":  (921600,  ASPECT_RATIO_720P),
    "900p":  (1440000, ASPECT_RATIO_900P),
    "1024":  (1048576, ASPECT_RATIO_1024),
    "1080p": (2073600, ASPECT_RATIO_1080P),
}


def get_num_pixels(name: str) -> int:
    """Return the nominal pixel count for a resolution name."""
    return ASPECT_RATIOS[name][0]


def get_closest_ratio(height: float, width: float, ratios: dict) -> str:
    """Find the aspect-ratio key closest to height/width."""
    ar = height / width
    return min(ratios.keys(), key=lambda r: abs(float(r) - ar))


# =====================================================================
# Default bucket configurations
# =====================================================================

BUCKET_CONFIG_DEFAULT = {
    "144p": {1: (1.0, 128), 16: (1.0, 8), 32: ((1.0, 0.33), 4), 64: ((1.0, 0.1), 2), 128: ((1.0, 0.1), 1)},
    "256":  {1: (0.4, 64), 16: (0.5, 4), 32: ((0.5, 0.33), 2), 64: ((0.5, 0.1), 1)},
    "240p": {1: (0.3, 64), 16: (0.4, 4), 32: ((0.4, 0.33), 2), 64: ((0.4, 0.1), 1)},
    "360p": {1: (0.2, 32), 16: (0.15, 2), 32: ((0.15, 0.33), 1)},
    "512":  {1: (0.1, 32)},
    "480p": {1: (0.1, 16)},
    "720p": {1: (0.05, 8)},
    "1024": {1: (0.05, 8)},
    "1080p": {1: (0.1, 1)},
}

BUCKET_CONFIG_MULTIVIEW = {
    "144p": {1: (1.0, 128), 6: (1.0, 16), 16: (1.0, 8), 32: ((1.0, 0.33), 4), 64: ((1.0, 0.1), 2), 128: ((1.0, 0.1), 1)},
    "256":  {1: (0.4, 64), 6: (0.5, 8), 16: (0.5, 4), 32: ((0.5, 0.33), 2), 64: ((0.5, 0.1), 1)},
    "240p": {1: (0.3, 64), 6: (0.4, 8), 16: (0.4, 4), 32: ((0.4, 0.33), 2), 64: ((0.4, 0.1), 1)},
    "360p": {1: (0.5, 32), 6: (0.15, 4), 16: ((0.3, 0.5), 2), 32: ((0.3, 1.0), 1)},
    "512":  {1: (0.4, 32), 6: (0.15, 4), 16: ((0.2, 0.4), 2), 32: ((0.2, 1.0), 1)},
    "480p": {1: (0.5, 16), 6: (0.2, 2), 16: (0.2, 1)},
    "720p": {1: (0.1, 8), 6: (0.03, 1)},
    "900p": {1: (0.1, 8), 6: (0.02, 1)},
    "1024": {1: (0.05, 8)},
    "1080p": {1: (0.1, 1)},
}

BUCKET_CONFIG_IMAGE_ONLY = {
    "144p": {1: (1.0, 128)},
    "256":  {1: (0.6, 64)},
    "240p": {1: (0.6, 64)},
    "360p": {1: (0.4, 32)},
    "512":  {1: (0.4, 32)},
    "480p": {1: (0.2, 16)},
    "720p": {1: (0.2, 8)},
    "900p": {1: (0.1, 4)},
}


# =====================================================================
# Bucket
# =====================================================================

class Bucket:
    """Manages resolution x frame-count buckets with probabilistic assignment.

    Each bucket is identified by a tuple ``(hw_name, t_id, ar_id)``.
    """

    def __init__(self, bucket_config: dict):
        for key in bucket_config:
            assert key in ASPECT_RATIOS, f"Aspect ratio '{key}' not found."

        bucket_probs = OrderedDict()
        bucket_bs = OrderedDict()
        bucket_names = sorted(bucket_config.keys(), key=lambda x: ASPECT_RATIOS[x][0], reverse=True)

        for key in bucket_names:
            time_keys = sorted(bucket_config[key].keys(), key=lambda x: x, reverse=True)
            bucket_probs[key] = OrderedDict({k: bucket_config[key][k][0] for k in time_keys})
            bucket_bs[key] = OrderedDict({k: bucket_config[key][k][1] for k in time_keys})

        num_bucket = 0
        hw_criteria, t_criteria, ar_criteria = {}, {}, {}
        bucket_id = OrderedDict()
        bucket_id_cnt = 0

        for hw_name, time_probs in bucket_probs.items():
            hw_criteria[hw_name] = ASPECT_RATIOS[hw_name][0]
            t_criteria[hw_name] = {}
            ar_criteria[hw_name] = {}
            bucket_id[hw_name] = {}
            for t_key in time_probs:
                t_criteria[hw_name][t_key] = t_key
                bucket_id[hw_name][t_key] = bucket_id_cnt
                bucket_id_cnt += 1
                ar_criteria[hw_name][t_key] = {}
                for ar_key, ar_hw in ASPECT_RATIOS[hw_name][1].items():
                    ar_criteria[hw_name][t_key][ar_key] = ar_hw
                    num_bucket += 1

        self.bucket_probs = bucket_probs
        self.bucket_bs = bucket_bs
        self.bucket_id = bucket_id
        self.hw_criteria = hw_criteria
        self.t_criteria = t_criteria
        self.ar_criteria = ar_criteria
        self.num_bucket = num_bucket

    def get_bucket_id(self, T: int, H: int, W: int,
                      frame_interval: int = 1, seed: Optional[int] = None):
        """Assign a sample to a bucket. Returns None if no bucket fits."""
        resolution = H * W
        approx = 0.8

        for hw_id, t_probs in self.bucket_probs.items():
            if resolution < self.hw_criteria[hw_id] * approx:
                continue

            if T == 1:
                if 1 in t_probs:
                    rng = np.random.default_rng(seed + self.bucket_id[hw_id][1])
                    if rng.random() < t_probs[1]:
                        ar_id = get_closest_ratio(H, W, self.ar_criteria[hw_id][1])
                        return hw_id, 1, ar_id
                continue

            t_found = False
            for t_id, prob in t_probs.items():
                rng = np.random.default_rng(seed + self.bucket_id[hw_id][t_id])
                if isinstance(prob, tuple):
                    if rng.random() > prob[1]:
                        continue
                if T > t_id * frame_interval and t_id != 1:
                    t_found = True
                    break

            if not t_found:
                continue

            final_prob = prob[0] if isinstance(prob, tuple) else prob
            if final_prob >= 1 or rng.random() < final_prob:
                ar_id = get_closest_ratio(H, W, self.ar_criteria[hw_id][t_id])
                return hw_id, t_id, ar_id

        return None

    def get_thw(self, bucket_id) -> Tuple[int, int, int]:
        """Return (T, H, W) for a bucket id tuple."""
        assert len(bucket_id) == 3
        T = self.t_criteria[bucket_id[0]][bucket_id[1]]
        H, W = self.ar_criteria[bucket_id[0]][bucket_id[1]][bucket_id[2]]
        return T, H, W

    def get_batch_size(self, bucket_id) -> int:
        """Return batch size for a bucket. Accepts 2- or 3-element tuple."""
        if len(bucket_id) == 3:
            return self.bucket_bs[bucket_id[0]][bucket_id[1]]
        return self.bucket_bs[bucket_id[0]][bucket_id[1]]

    def __len__(self) -> int:
        return self.num_bucket


# =====================================================================
# Sampler
# =====================================================================

class VariableVideoBatchSampler(DistributedSampler):
    """Distributed batch sampler that groups samples into resolution buckets.

    Each yielded batch is a list of string-encoded indices ``'<idx>-<T>-<H>-<W>'``.
    """

    def __init__(
        self,
        dataset: Dataset,
        bucket_config: dict,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        verbose: bool = False,
        num_bucket_build_workers: int = 1,
    ):
        super().__init__(dataset=dataset, num_replicas=num_replicas,
                         rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        self.dataset = dataset
        self.bucket = Bucket(bucket_config)
        self.verbose = verbose
        self.last_micro_batch_access_index = 0
        self.approximate_num_batch = None
        self._cached_bucket_sample_dict = None
        self.num_bucket_build_workers = num_bucket_build_workers

    def group_by_bucket(self) -> dict:
        """Assign each dataset sample to its bucket."""
        bucket_sample_dict = OrderedDict()
        for i in range(len(self.dataset)):
            data_entry = self.dataset.data[i]
            num_frames = len(data_entry)
            bucket_id = self.bucket.get_bucket_id(
                num_frames, 900, 1600,
                self.dataset.frame_interval,
                self.seed + self.epoch + i * self.bucket.num_bucket,
            )
            if bucket_id is None:
                continue
            bucket_sample_dict.setdefault(bucket_id, []).append(i)
        return bucket_sample_dict

    def get_num_batch(self) -> int:
        """Compute total number of batches across all buckets."""
        bucket_sample_dict = self.group_by_bucket()
        self._cached_bucket_sample_dict = bucket_sample_dict
        if self.verbose:
            self._print_bucket_info(bucket_sample_dict)
        return self.approximate_num_batch

    def __iter__(self) -> Iterator[List[str]]:
        if self._cached_bucket_sample_dict is not None:
            bucket_sample_dict = self._cached_bucket_sample_dict
            self._cached_bucket_sample_dict = None
        else:
            bucket_sample_dict = self.group_by_bucket()
            if self.verbose:
                self._print_bucket_info(bucket_sample_dict)

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        bucket_micro_batch_count = OrderedDict()
        bucket_last_consumed = OrderedDict()

        for bucket_id, data_list in bucket_sample_dict.items():
            bs = self.bucket.get_batch_size(bucket_id)
            remainder = len(data_list) % bs
            if remainder > 0:
                if not self.drop_last:
                    data_list += data_list[:bs - remainder]
                else:
                    data_list = data_list[:-remainder]
            bucket_sample_dict[bucket_id] = data_list

            if self.shuffle:
                indices = torch.randperm(len(data_list), generator=g).tolist()
                data_list = [data_list[i] for i in indices]
                bucket_sample_dict[bucket_id] = data_list

            bucket_micro_batch_count[bucket_id] = len(data_list) // bs

        # Build access order
        access_order = []
        for bid, count in bucket_micro_batch_count.items():
            access_order.extend([bid] * count)

        if self.shuffle:
            perm = torch.randperm(len(access_order), generator=g).tolist()
            access_order = [access_order[i] for i in perm]

        # Make divisible by num_replicas
        remainder = len(access_order) % self.num_replicas
        if remainder > 0:
            if self.drop_last:
                access_order = access_order[:len(access_order) - remainder]
            else:
                access_order += access_order[:self.num_replicas - remainder]

        num_iters = len(access_order) // self.num_replicas
        start_iter = self.last_micro_batch_access_index // self.num_replicas
        self.last_micro_batch_access_index = start_iter * self.num_replicas

        for i in range(self.last_micro_batch_access_index):
            bid = access_order[i]
            bs = self.bucket.get_batch_size(bid)
            bucket_last_consumed[bid] = bucket_last_consumed.get(bid, 0) + bs

        for i in range(start_iter, num_iters):
            access_list = access_order[i * self.num_replicas:(i + 1) * self.num_replicas]
            self.last_micro_batch_access_index += self.num_replicas

            boundaries = []
            for bid in access_list:
                bs = self.bucket.get_batch_size(bid)
                last = bucket_last_consumed.get(bid, 0)
                boundaries.append((last, last + bs))
                bucket_last_consumed[bid] = bucket_last_consumed.get(bid, 0) + bs

            bid = access_list[self.rank]
            lo, hi = boundaries[self.rank]
            batch = bucket_sample_dict[bid][lo:hi]

            real_t, real_h, real_w = self.bucket.get_thw(bid)
            yield [f"{idx}-{real_t}-{real_h}-{real_w}" for idx in batch]

        self.reset()

    def _print_bucket_info(self, bucket_sample_dict: dict) -> None:
        total_samples, total_batch = 0, 0
        num_hwt_dict = defaultdict(lambda: [0, 0])

        for k, v in bucket_sample_dict.items():
            size = len(v)
            nb = size // self.bucket.get_batch_size(k[:-1])
            total_samples += size
            total_batch += nb
            num_hwt_dict[k[:-1]][0] += size
            num_hwt_dict[k[:-1]][1] += nb

        if dist.is_initialized() and dist.get_rank() == 0 and self.verbose:
            logger.info("Bucket info [#sample, #batch] by HxWxT:\n%s",
                        pformat(dict(num_hwt_dict), sort_dicts=False))
            logger.info("#batches: %d, #samples: %d, #non-empty buckets: %d",
                        total_batch, total_samples, len(bucket_sample_dict))

        self.approximate_num_batch = total_batch

    def reset(self):
        self.last_micro_batch_access_index = 0

    def __len__(self) -> int:
        return self.get_num_batch() // (dist.get_world_size() if dist.is_initialized() else 1)

    def state_dict(self, num_steps: int) -> dict:
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "last_micro_batch_access_index": num_steps * self.num_replicas,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)


# =====================================================================
# Factory
# =====================================================================

def _seed_worker(seed):
    def _fn(worker_id):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
    return _fn


def prepare_dataloader(
    dataset: Dataset,
    bucket_config: dict,
    process_group: ProcessGroup,
    shuffle: bool = False,
    seed: int = 1024,
    drop_last: bool = False,
    pin_memory: bool = False,
    num_workers: int = 0,
    num_bucket_build_workers: int = 1,
    prefetch_factor: Optional[int] = None,
    **kwargs,
) -> Tuple[DataLoader, VariableVideoBatchSampler]:
    """Build a DataLoader with bucket-based variable-resolution batch sampling.

    Returns:
        A ``(DataLoader, VariableVideoBatchSampler)`` tuple.
    """
    batch_sampler = VariableVideoBatchSampler(
        dataset,
        bucket_config,
        num_replicas=process_group.size(),
        rank=process_group.rank(),
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
        verbose=True,
        num_bucket_build_workers=num_bucket_build_workers,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        worker_init_fn=_seed_worker(seed),
        pin_memory=pin_memory,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        **kwargs,
    )
    return loader, batch_sampler
