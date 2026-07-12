# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Optional MoE routing and token-dropping counters.

The hook is intentionally opt-in because it adds small GPU reductions on the
router path. Enable it with ``BRIDGE_MOE_TOKEN_DROP_METRICS=1`` when routing
health must be measured. Capacity-limited routing records the map before and
after token dropping; dropless routing records the returned map as both routed
and kept so an exact zero-drop observation is still emitted.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any

import torch

from megatron.bridge.utils.common_utils import get_rank_safe, print_rank_0


logger = logging.getLogger(__name__)

METRIC_NAMES = (
    "routed_assignments",
    "kept_assignments",
    "dropped_assignments",
    "tokens_with_zero_kept_experts",
    "tokens_with_1_kept_experts",
    "tokens_with_2_kept_experts",
    "tokens_with_3_kept_experts",
    "tokens_with_4_kept_experts",
)

_COUNTS: dict[int, torch.Tensor] = {}
_INSTALLED = False
_ORIGINAL_APPLY_ROUTER_TOKEN_DROPPING = None
_CURRENT_LAYER_STACK: list[int] = []


def _env_enabled() -> bool:
    return str(os.environ.get("BRIDGE_MOE_TOKEN_DROP_METRICS", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _router_layer_index() -> int | None:
    if _CURRENT_LAYER_STACK:
        return _CURRENT_LAYER_STACK[-1]
    frame = inspect.currentframe()
    try:
        caller = frame.f_back.f_back if frame is not None and frame.f_back is not None else None
        router = caller.f_locals.get("self") if caller is not None else None
        layer_number = getattr(router, "layer_number", None)
        if layer_number is None:
            return None
        config = getattr(router, "config", None)
        if getattr(router, "is_mtp_layer", False) and config is not None:
            layer_number += int(getattr(config, "num_layers", 0) or 0)
        return max(0, int(layer_number) - 1)
    finally:
        del frame


def _record_counts(
    *,
    routing_map: torch.Tensor,
    final_map: torch.Tensor,
    layer_index: int,
) -> None:
    with torch.no_grad():
        original_map = routing_map.bool()
        kept_map = torch.logical_and(original_map, final_map.bool())
        kept_per_token = kept_map.sum(dim=1)

        routed = original_map.sum(dtype=torch.float64)
        kept = kept_map.sum(dtype=torch.float64)
        dropped = routed - kept
        values = torch.stack(
            (
                routed,
                kept,
                dropped,
                (kept_per_token == 0).sum(dtype=torch.float64),
                (kept_per_token == 1).sum(dtype=torch.float64),
                (kept_per_token == 2).sum(dtype=torch.float64),
                (kept_per_token == 3).sum(dtype=torch.float64),
                (kept_per_token == 4).sum(dtype=torch.float64),
            )
        )

        if layer_index not in _COUNTS:
            _COUNTS[layer_index] = torch.zeros_like(values)
        _COUNTS[layer_index].add_(values)


def install_moe_token_drop_metric_hooks(enabled: bool | None = None) -> bool:
    """Install an opt-in wrapper around Megatron-Core router token dropping."""

    global _INSTALLED, _ORIGINAL_APPLY_ROUTER_TOKEN_DROPPING
    if enabled is None:
        enabled = _env_enabled()
    if not enabled:
        return False
    if _INSTALLED:
        return True

    try:
        import megatron.core.transformer.moe.router as router_mod
    except Exception as exc:  # pragma: no cover - defensive for non-MoE installs.
        logger.warning("Could not install MoE token-drop metrics hook: %s", exc)
        return False

    original = router_mod.apply_router_token_dropping
    if getattr(original, "_bridge_moe_token_drop_metrics_hook", False):
        _INSTALLED = True
        return True

    def wrapped_apply_router_token_dropping(*args: Any, **kwargs: Any):
        final_probs, final_map = original(*args, **kwargs)
        try:
            routing_map = kwargs.get("routing_map")
            if routing_map is None and len(args) >= 2:
                routing_map = args[1]
            if routing_map is not None:
                layer_index = _router_layer_index()
                if layer_index is not None:
                    _record_counts(
                        routing_map=routing_map,
                        final_map=final_map,
                        layer_index=layer_index,
                    )
        except Exception as exc:  # Keep metric collection from breaking training.
            logger.debug("MoE token-drop metric collection failed: %s", exc)
        return final_probs, final_map

    wrapped_apply_router_token_dropping._bridge_moe_token_drop_metrics_hook = True
    router_mod.apply_router_token_dropping = wrapped_apply_router_token_dropping
    _ORIGINAL_APPLY_ROUTER_TOKEN_DROPPING = original

    original_routing = router_mod.TopKRouter.routing
    if not getattr(original_routing, "_bridge_moe_token_drop_metrics_context_hook", False):

        def wrapped_topk_routing(self: Any, *args: Any, **kwargs: Any):
            layer_number = getattr(self, "layer_number", None)
            layer_index = None
            if layer_number is not None:
                config = getattr(self, "config", None)
                if getattr(self, "is_mtp_layer", False) and config is not None:
                    layer_number += int(getattr(config, "num_layers", 0) or 0)
                layer_index = max(0, int(layer_number) - 1)
            if layer_index is None:
                return original_routing(self, *args, **kwargs)
            _CURRENT_LAYER_STACK.append(layer_index)
            try:
                result = original_routing(self, *args, **kwargs)
                # Megatron-Core calls apply_router_token_dropping only when a
                # capacity factor is configured. In the dropless path the
                # returned routing map is therefore simultaneously the routed
                # and kept map. Record it here so zero-drop evidence exists,
                # while leaving capacity-limited accounting solely to the
                # apply_router_token_dropping wrapper above to avoid counting
                # the same forward twice.
                try:
                    config = getattr(self, "config", None)
                    if (
                        getattr(config, "moe_expert_capacity_factor", None) is None
                        and isinstance(result, tuple)
                        and len(result) == 2
                        and isinstance(result[1], torch.Tensor)
                    ):
                        _record_counts(
                            routing_map=result[1],
                            final_map=result[1],
                            layer_index=layer_index,
                        )
                except Exception as exc:  # Keep metric collection from breaking training.
                    logger.debug("Dropless MoE token-drop metric collection failed: %s", exc)
                return result
            finally:
                _CURRENT_LAYER_STACK.pop()

        wrapped_topk_routing._bridge_moe_token_drop_metrics_context_hook = True
        router_mod.TopKRouter.routing = wrapped_topk_routing

    _INSTALLED = True
    print_rank_0("Enabled Bridge MoE token-drop metrics hook.")
    return True


def _distributed_ready() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _metric_matrix() -> torch.Tensor | None:
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    local_layers = max(_COUNTS.keys(), default=-1) + 1
    if _distributed_ready():
        layer_count = torch.tensor([local_layers], device=device, dtype=torch.long)
        torch.distributed.all_reduce(layer_count, op=torch.distributed.ReduceOp.MAX)
        total_layers = int(layer_count.item())
    else:
        total_layers = local_layers
    if total_layers <= 0:
        return None

    matrix = torch.zeros((total_layers, len(METRIC_NAMES)), device=device, dtype=torch.float64)
    for layer_index, values in _COUNTS.items():
        if layer_index < total_layers:
            matrix[layer_index].add_(values.to(device=device, dtype=torch.float64))
    if _distributed_ready():
        torch.distributed.all_reduce(matrix, op=torch.distributed.ReduceOp.SUM)
    return matrix


def _log_scalar(name: str, value: float, iteration: int, writer: Any, wandb_writer: Any) -> None:
    if writer is not None:
        writer.add_scalar(name, value, iteration)
    if wandb_writer is not None:
        wandb_writer.log({name: value}, iteration)


def flush_moe_token_drop_metrics(
    *,
    iteration: int,
    writer: Any = None,
    wandb_writer: Any = None,
    per_layer_logging: bool = True,
) -> None:
    """Reduce and log token-dropping counters collected since the last flush."""

    if not _INSTALLED:
        return
    matrix = _metric_matrix()
    _COUNTS.clear()
    if matrix is None or get_rank_safe() != 0:
        return

    totals = matrix.sum(dim=0)
    metrics = {name: float(totals[i].item()) for i, name in enumerate(METRIC_NAMES)}
    routed = max(metrics["routed_assignments"], 1.0)
    metrics["dropped_assignment_rate"] = metrics["dropped_assignments"] / routed

    for name, value in metrics.items():
        _log_scalar(f"moe_token_drop/{name}", value, iteration, writer, wandb_writer)

    print_rank_0(
        "[moe_token_drop] "
        f"iteration {iteration} | "
        f"routed_assignments: {metrics['routed_assignments']:.0f} | "
        f"kept_assignments: {metrics['kept_assignments']:.0f} | "
        f"dropped_assignments: {metrics['dropped_assignments']:.0f} | "
        f"dropped_assignment_rate: {metrics['dropped_assignment_rate']:.6f} | "
        f"tokens_with_zero_kept_experts: {metrics['tokens_with_zero_kept_experts']:.0f} | "
        f"tokens_with_1_kept_experts: {metrics['tokens_with_1_kept_experts']:.0f} | "
        f"tokens_with_2_kept_experts: {metrics['tokens_with_2_kept_experts']:.0f} | "
        f"tokens_with_3_kept_experts: {metrics['tokens_with_3_kept_experts']:.0f} | "
        f"tokens_with_4_kept_experts: {metrics['tokens_with_4_kept_experts']:.0f}"
    )

    if not per_layer_logging:
        return

    for layer_index, row in enumerate(matrix):
        layer_metrics = {name: float(row[i].item()) for i, name in enumerate(METRIC_NAMES)}
        layer_routed = max(layer_metrics["routed_assignments"], 1.0)
        layer_metrics["dropped_assignment_rate"] = layer_metrics["dropped_assignments"] / layer_routed
        for name, value in layer_metrics.items():
            _log_scalar(
                f"moe_token_drop/layer_{layer_index}/{name}",
                value,
                iteration,
                writer,
                wandb_writer,
            )
