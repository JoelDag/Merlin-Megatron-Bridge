# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Context-parallel compatibility helpers for Megatron-Core API drift."""

from __future__ import annotations

import inspect
from typing import Any

import torch
from megatron.core.utils import get_batch_on_this_cp_rank


_GET_BATCH_ON_THIS_CP_RANK_PARAMS = inspect.signature(get_batch_on_this_cp_rank).parameters
_SUPPORTS_IS_HYBRID_CP = "is_hybrid_cp" in _GET_BATCH_ON_THIS_CP_RANK_PARAMS


def get_batch_on_this_cp_rank_compat(
    batch: dict[str, Any],
    *,
    cp_group: torch.distributed.ProcessGroup | None = None,
    is_hybrid_cp: bool = False,
    hybrid_cp_group_func: Any | None = None,
) -> dict[str, Any]:
    """Call Megatron-Core CP batch slicing across old and new signatures.

    Megatron-Core r0.18 adds a required ``is_hybrid_cp`` argument. Bridge
    pretraining paths use non-hybrid CP, so pass it explicitly when supported.
    """

    if _SUPPORTS_IS_HYBRID_CP:
        return get_batch_on_this_cp_rank(
            batch,
            is_hybrid_cp=is_hybrid_cp,
            cp_group=cp_group,
            hybrid_cp_group_func=hybrid_cp_group_func,
        )

    if is_hybrid_cp:
        raise RuntimeError("Hybrid context parallelism requires a Megatron-Core CP slicing API with is_hybrid_cp")
    if hybrid_cp_group_func is not None:
        raise RuntimeError("hybrid_cp_group_func is only supported by the newer Megatron-Core CP slicing API")
    return get_batch_on_this_cp_rank(batch, cp_group=cp_group)
