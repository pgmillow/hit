import pytest
import torch

from pi05.train.engine.trainer import Pi05LoraTrainer


class _FakeAccelerator:
    device = torch.device("cpu")
    num_processes = 4

    def gather(self, payload: torch.Tensor) -> torch.Tensor:
        assert payload.tolist() == [2.0, 1.0]
        return torch.tensor([2.0, 1.0, 6.0, 2.0, 3.0, 1.0, 9.0, 2.0])

    def gather_for_metrics(self, payload: torch.Tensor) -> torch.Tensor:
        raise AssertionError("fixed-size rank statistics must not be gathered as batch metrics")


def test_resolve_accumulated_loss_gathers_fixed_size_rank_statistics() -> None:
    result = Pi05LoraTrainer._resolve_accumulated_loss(
        _FakeAccelerator(),
        loss_sum=torch.tensor(2.0),
        loss_count=1,
    )

    assert result == pytest.approx(20.0 / 6.0)
