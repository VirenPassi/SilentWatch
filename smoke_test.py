"""CPU smoke test for the SilentWatch dual-head video architecture.

This script intentionally uses randomly generated video clips instead of a
dataset. It validates the complete forward and backward tensor flow:

    (B, T, C, H, W) -> MobileNetV2 -> (B, T, D) -> LSTM -> dual heads

Run from the Silent_watch-main directory with:

    python smoke_test.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v2


class SilentWatch(nn.Module):
    """MobileNetV2 spatial encoder followed by an LSTM and two prediction heads."""

    def __init__(
        self,
        num_classes: int = 5,
        hidden_dim: int = 128,
        num_lstm_layers: int = 1,
    ) -> None:
        super().__init__()

        # weights=None is the modern torchvision equivalent of pretrained=False.
        backbone = mobilenet_v2(weights=None)
        self.spatial_encoder = backbone.features
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = backbone.last_channel  # 1280 for MobileNetV2

        self.temporal_encoder = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        # The classification head returns raw logits for CrossEntropyLoss.
        self.classification_head = nn.Linear(hidden_dim, num_classes)
        # The regression head predicts a continuous frame index.
        self.regression_head = nn.Linear(hidden_dim, 1)

    def forward(self, clips: Tensor) -> tuple[Tensor, Tensor]:
        """Return class logits and ToA predictions for clips shaped (B, T, C, H, W)."""
        if clips.ndim != 5:
            raise ValueError(
                f"Expected a 5D clip tensor (B, T, C, H, W), got {clips.shape}"
            )

        batch_size, num_frames, channels, height, width = clips.shape

        # MobileNetV2 expects (N, C, H, W), so process every frame as one image.
        frames = clips.reshape(batch_size * num_frames, channels, height, width)
        frame_features = self.spatial_encoder(frames)
        frame_features = self.global_average_pool(frame_features)
        frame_features = frame_features.flatten(start_dim=1)

        # Restore the temporal dimension: (B*T, D) -> (B, T, D).
        sequence_features = frame_features.reshape(
            batch_size, num_frames, self.feature_dim
        )

        # The final LSTM hidden state summarizes the full clip.
        _, (hidden_state, _) = self.temporal_encoder(sequence_features)
        context = hidden_state[-1]

        class_logits = self.classification_head(context)
        toa_prediction = self.regression_head(context).squeeze(-1)
        return class_logits, toa_prediction


class DummyDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """A deterministic-size random dataset for the smoke test."""

    def __init__(
        self,
        num_samples: int = 4,
        num_frames: int = 16,
        num_classes: int = 5,
    ) -> None:
        self.clips = torch.randn(num_samples, num_frames, 3, 224, 224)
        self.class_targets = torch.randint(0, num_classes, (num_samples,))
        self.toa_targets = torch.rand(num_samples) * (num_frames - 1)

    def __len__(self) -> int:
        return self.clips.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return (
            self.clips[index],
            self.class_targets[index],
            self.toa_targets[index],
        )


def run_smoke_test() -> None:
    """Run one CPU training step and assert the expected tensor flow."""
    torch.manual_seed(42)
    device = torch.device("cpu")

    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = SilentWatch(num_classes=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    classification_loss = nn.CrossEntropyLoss()
    regression_loss = nn.MSELoss()

    model.train()
    for clips, class_targets, toa_targets in loader:
        clips = clips.to(device)
        class_targets = class_targets.to(device)
        toa_targets = toa_targets.to(device)

        class_logits, toa_prediction = model(clips)
        assert class_logits.shape == (4, 5)
        assert toa_prediction.shape == (4,)

        total_loss = (
            1.0 * classification_loss(class_logits, class_targets)
            + 2.0 * regression_loss(toa_prediction, toa_targets)
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(
            f"Smoke test passed: clips={tuple(clips.shape)}, "
            f"logits={tuple(class_logits.shape)}, "
            f"toa={tuple(toa_prediction.shape)}, loss={total_loss.item():.4f}"
        )


if __name__ == "__main__":
    run_smoke_test()
