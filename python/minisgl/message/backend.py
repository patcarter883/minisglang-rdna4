from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    pass


@dataclass
class MMImage(BaseBackendMsg):
    """One image of a request, preprocessed on the tokenizer worker (see tokenizer/vision.py).

    `pixels` is the resized image as raw uint8 PATCHES — (pw*ph, patch*patch*3) row-major, patches in
    raster order — exactly the values the reference processor produces before its /255 rescale, so the
    GPU's `vision_patch_in` reproduces the reference input bit for bit from 1/4 of the fp32 bytes.
    `offset`/`length` locate the image's soft tokens in the request's input_ids, where they are filled
    with a content-hash pad id (>= vocab) so the radix cache can never match two different images."""
    pixels: bytes
    pw: int
    ph: int
    offset: int
    length: int
    content_hash: int


@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams
    mm_images: List[MMImage] | None = None


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int
