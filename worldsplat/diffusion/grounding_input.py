"""Grounding tokenizer input: prepares structured bounding box inputs for the grounding network."""

import torch


class GroundingNetInput:
    """Prepares and caches grounding network inputs from a data batch.

    Extracts 2D bounding boxes, heading, instance IDs, masks, and positive
    (text) embeddings from a batch dict. Also provides null inputs for
    classifier-free guidance during training (random drop) or inference.
    """

    def __init__(self):
        self._is_set = False

    def prepare(self, batch: dict) -> dict:
        """Extract grounding fields from a data batch.

        Args:
            batch: Dataset output dict containing:
                - boxes_2d: [B, N, 4] normalized 2D bounding boxes
                - masks: [B, N] binary validity mask
                - heading: [B, N] heading angles in degrees
                - instance_id: [B, N] instance ID scalars
                - positive_embedding: [B, N, D] text/positive embeddings

        Returns:
            Dict with keys: boxes_2d, masks, heading, instance_id, positive_embeddings.
        """
        self._is_set = True

        boxes_2d = batch["boxes_2d"]
        masks = batch["masks"]
        heading = batch["heading"]
        instance_id = batch["instance_id"]
        positive_embeddings = batch["positive_embedding"]

        self.batch_size, self.max_box, self.in_dim = positive_embeddings.shape
        self.device = positive_embeddings.device
        self.dtype = positive_embeddings.dtype
        self.heading_dtype = heading.dtype
        self.instance_id_dtype = instance_id.dtype

        return {
            "boxes_2d": boxes_2d,
            "masks": masks,
            "heading": heading,
            "instance_id": instance_id,
            "positive_embeddings": positive_embeddings,
        }

    def get_null_input(
        self,
        batch_size: int = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> dict:
        """Generate all-zeros null input for classifier-free guidance.

        Uses cached shape/device/dtype from the last `prepare()` call as defaults.

        Args:
            batch_size: Override batch size (defaults to last prepared batch size).
            device: Override device (defaults to last prepared device).
            dtype: Override dtype (defaults to last prepared dtype).

        Returns:
            Dict with the same keys as `prepare()`, all filled with zeros.
        """
        assert self._is_set, "prepare() must be called before get_null_input()"
        batch_size = self.batch_size if batch_size is None else batch_size
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype

        boxes_2d = torch.zeros(batch_size, self.max_box, 4, dtype=dtype, device=device)
        masks = torch.zeros(batch_size, self.max_box, dtype=dtype, device=device)
        heading = torch.zeros(batch_size, self.max_box, dtype=self.heading_dtype, device=device)
        instance_id = torch.zeros(batch_size, self.max_box, dtype=self.instance_id_dtype, device=device)
        positive_embeddings = torch.zeros(batch_size, self.max_box, self.in_dim, dtype=dtype, device=device)

        return {
            "boxes_2d": boxes_2d,
            "masks": masks,
            "heading": heading,
            "instance_id": instance_id,
            "positive_embeddings": positive_embeddings,
        }
