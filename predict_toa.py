"""Print dual-head predictions for a few preprocessed metadata rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_loader import get_dataloader  # noqa: E402
from models.model import SilentWatchModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    with (Path(__file__).parent / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SilentWatchModel(
        num_classes=config["model"]["num_classes"],
        temporal_module=config["model"]["temporal_module"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    loader = get_dataloader(
        args.metadata,
        batch_size=min(args.limit, config["training"]["batch_size"]),
        seq_length=config["dataset"]["seq_length"],
        shuffle=False,
    )
    seen = 0
    with torch.no_grad():
        for frames, labels, toa_targets in loader:
            logits, toa_predictions = model(frames.to(device))
            class_predictions = logits.argmax(dim=1).cpu()
            toa_predictions = toa_predictions.cpu()
            for batch_index, (class_id, toa) in enumerate(
                zip(class_predictions, toa_predictions)
            ):
                print(
                    f"sample={seen} predicted_class={class_id.item()} "
                    f"predicted_toa_frame={toa.item():.3f} "
                    f"target_toa_frame={toa_targets[batch_index].item():.3f}"
                )
                seen += 1
                if seen >= args.limit:
                    return


if __name__ == "__main__":
    main()
