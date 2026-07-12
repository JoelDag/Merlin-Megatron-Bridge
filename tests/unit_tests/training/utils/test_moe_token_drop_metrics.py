import re
from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.training.utils import moe_token_drop_metrics


@pytest.fixture
def isolated_router_hooks(monkeypatch):
    from megatron.core.transformer.moe import router as router_module

    # Register both functions with monkeypatch before the installer replaces
    # them internally, so every test restores an unwrapped Megatron-Core module.
    monkeypatch.setattr(
        router_module,
        "apply_router_token_dropping",
        router_module.apply_router_token_dropping,
    )
    monkeypatch.setattr(router_module.TopKRouter, "routing", router_module.TopKRouter.routing)
    moe_token_drop_metrics._COUNTS.clear()
    moe_token_drop_metrics._CURRENT_LAYER_STACK.clear()
    monkeypatch.setattr(moe_token_drop_metrics, "_INSTALLED", False)
    monkeypatch.setattr(moe_token_drop_metrics, "_ORIGINAL_APPLY_ROUTER_TOKEN_DROPPING", None)
    yield router_module
    moe_token_drop_metrics._COUNTS.clear()
    moe_token_drop_metrics._CURRENT_LAYER_STACK.clear()


def _router(*, capacity_factor: float | None, layer_number: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(moe_expert_capacity_factor=capacity_factor, num_layers=4),
        is_mtp_layer=False,
        layer_number=layer_number,
    )


@pytest.mark.unit
def test_dropless_topk_routing_records_exact_zero_drop(isolated_router_hooks, monkeypatch):
    router_module = isolated_router_hooks
    routing_map = torch.tensor(
        [
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, False],
        ]
    )

    def fake_routing(_router, logits, padding_mask=None):
        del padding_mask
        return torch.ones_like(logits), routing_map

    monkeypatch.setattr(router_module.TopKRouter, "routing", fake_routing)
    assert moe_token_drop_metrics.install_moe_token_drop_metric_hooks(enabled=True)

    _probabilities, observed_map = router_module.TopKRouter.routing(_router(capacity_factor=None), torch.zeros((3, 4)))

    assert torch.equal(observed_map, routing_map)
    assert set(moe_token_drop_metrics._COUNTS) == {1}
    assert moe_token_drop_metrics._COUNTS[1].tolist() == [3.0, 3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]


@pytest.mark.unit
def test_capacity_drop_path_is_not_double_counted(isolated_router_hooks, monkeypatch):
    router_module = isolated_router_hooks
    routing_map = torch.tensor(
        [
            [True, False],
            [False, True],
            [True, False],
            [False, True],
        ]
    )

    def fake_apply_router_token_dropping(probs, original_map, **_kwargs):
        final_map = original_map.clone()
        final_map[2, 0] = False
        return probs, final_map

    def fake_routing(router, logits, padding_mask=None):
        del padding_mask
        return router_module.apply_router_token_dropping(
            torch.ones_like(logits),
            routing_map,
            router_topk=1,
            capacity_factor=router.config.moe_expert_capacity_factor,
            drop_policy="probs",
            pad_to_capacity=False,
        )

    monkeypatch.setattr(router_module, "apply_router_token_dropping", fake_apply_router_token_dropping)
    monkeypatch.setattr(router_module.TopKRouter, "routing", fake_routing)
    assert moe_token_drop_metrics.install_moe_token_drop_metric_hooks(enabled=True)

    _probabilities, final_map = router_module.TopKRouter.routing(_router(capacity_factor=0.5), torch.zeros((4, 2)))

    assert final_map.sum().item() == 3
    assert moe_token_drop_metrics._COUNTS[1].tolist() == [4.0, 3.0, 1.0, 1.0, 3.0, 0.0, 0.0, 0.0]


@pytest.mark.unit
def test_dropless_log_line_matches_training_health_parser(isolated_router_hooks, monkeypatch):
    router_module = isolated_router_hooks
    routing_map = torch.tensor([[True, False], [False, True]])

    def fake_routing(_router, logits, padding_mask=None):
        del padding_mask
        return torch.ones_like(logits), routing_map

    emitted: list[str] = []
    monkeypatch.setattr(router_module.TopKRouter, "routing", fake_routing)
    monkeypatch.setattr(moe_token_drop_metrics, "print_rank_0", emitted.append)
    monkeypatch.setattr(moe_token_drop_metrics, "get_rank_safe", lambda: 0)
    assert moe_token_drop_metrics.install_moe_token_drop_metric_hooks(enabled=True)
    router_module.TopKRouter.routing(_router(capacity_factor=None), torch.zeros((2, 2)))
    assert moe_token_drop_metrics._COUNTS[1].tolist() == [2.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]
    moe_token_drop_metrics.flush_moe_token_drop_metrics(iteration=7, per_layer_logging=False)

    line = next(value for value in emitted if value.startswith("[moe_token_drop]"))
    parser_pattern = re.compile(
        r"\[moe_token_drop\].*?dropped_assignment_rate:\s*(?P<drop_rate>[0-9.]+).*?"
        r"tokens_with_zero_kept_experts:\s*(?P<zero_kept>[0-9.]+)",
        re.IGNORECASE,
    )
    match = parser_pattern.search(line)
    assert match is not None
    assert float(match.group("drop_rate")) == 0.0
    assert float(match.group("zero_kept")) == 0.0
