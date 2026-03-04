"""
Weights & Biases logging helper.
"""

from __future__ import annotations


class WandbLogger:
    """Thin wrapper around the ``wandb`` Python SDK.

    Parameters
    ----------
    opt:
        Full config dict.  Must contain a ``wandb.project`` key.
    """

    def __init__(self, opt: dict) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "WandbLogger requires wandb.  Install it with: pip install wandb"
            ) from exc

        self._wandb = wandb
        if self._wandb.run is None:
            self._wandb.init(
                project=opt["wandb"]["project"],
                config=opt,
                dir="./experiments",
            )
        self.config = self._wandb.config

    def log_metrics(self, metrics: dict, commit: bool = True) -> None:
        """Log a dictionary of scalar metrics."""
        self._wandb.log(metrics, commit=commit)

    def log_image(self, key: str, images) -> None:
        """Log images."""
        self._wandb.log({key: [self._wandb.Image(img) for img in images]})
