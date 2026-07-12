import inspect

import torch

from megatron.bridge.training.eval import (
    _finalize_loss_dict,
    _loss_accumulator_dtype,
    evaluate,
)


def test_evaluate_sufficient_statistics_mode_preserves_float64_pair() -> None:
    statistics = torch.tensor([12.5, 5.0], dtype=torch.float64)
    losses = {"lm loss": statistics}

    result = _finalize_loss_dict(
        losses,
        return_loss_sufficient_statistics=True,
    )

    assert _loss_accumulator_dtype(return_loss_sufficient_statistics=True) == torch.float64
    assert result is losses
    assert result["lm loss"] is statistics
    assert result["lm loss"].dtype == torch.float64
    torch.testing.assert_close(result["lm loss"], torch.tensor([12.5, 5.0], dtype=torch.float64))


def test_evaluate_default_mode_keeps_legacy_average() -> None:
    losses = {"lm loss": torch.tensor([12.5, 5.0], dtype=torch.float32)}

    result = _finalize_loss_dict(
        losses,
        return_loss_sufficient_statistics=False,
    )

    assert inspect.signature(evaluate).parameters["return_loss_sufficient_statistics"].default is False
    assert _loss_accumulator_dtype(return_loss_sufficient_statistics=False) == torch.float32
    assert result is losses
    assert result["lm loss"].dtype == torch.float32
    torch.testing.assert_close(result["lm loss"], torch.tensor(2.5, dtype=torch.float32))
