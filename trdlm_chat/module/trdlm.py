import logging
from enum import StrEnum
from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from ..loss.aggregators import LossAggregator, LossOutput
from ..model.trm import TinyRecursiveModel
from ..utils.logger import LOGGER
from .learning import TrdlmLearningParams


class _Phase(StrEnum):
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"


class TRDLM(L.LightningModule):
    def __init__(
        self,
        model: TinyRecursiveModel,
        learning_params: TrdlmLearningParams,
        loss_aggregator: LossAggregator | None = None,
        logger: logging.Logger = LOGGER,
    ) -> None:
        super().__init__()

        self._model = model
        self._learning_params = learning_params
        self._loss_aggregator = loss_aggregator
        self._logger = logger

        # Lightning-specific setup
        self.automatic_optimization = False

    @property
    def model(self) -> TinyRecursiveModel:
        return self._model

    def log_loss(self, loss: LossOutput, phase: str) -> torch.Tensor:
        """
        Handles the loss logging (to Tensorboard).

        Args:
            loss (LossOutput): The loss output object containing individual losses.
            phase (str): The phase of the training (e.g., "train", "val").

        Returns:
            torch.Tensor: The total loss.
        """
        for name in loss.individual:
            log_name = f"{phase}\\{name.replace('_', ' ')}"
            self.log(
                log_name,
                loss.individual[name],
                batch_size=self._learning_params.device_batch_size,
                sync_dist=True,
            )
        self.log(
            f"{phase}\\total",
            loss.total,
            prog_bar=True,
            batch_size=self._learning_params.device_batch_size,
            sync_dist=True,
        )
        return loss.total

    def configure_optimizers(self) -> OptimizerLRScheduler:
        muon_params = self._model.core.get_muon_params()
        core_adamw_params = self._model.core.get_adamw_params()
        embedding_params = list(self._model.input_embedding.parameters())
        output_head_params = list(self._model.output_head.parameters())
        q_head_params = list(self._model.q_head.parameters())

        adamw_groups = [
            dict(
                params=core_adamw_params + output_head_params + q_head_params,
                lr=self._learning_params.adamw_non_embedding_lr,
            ),
            dict(params=embedding_params, lr=self._learning_params.adamw_embedding_lr),
        ]

        adamw_optimizer = torch.optim.AdamW(
            adamw_groups,
            betas=(0.8, 0.95),
            fused=True,
            eps=1e-10,
            weight_decay=self._learning_params.adamw_weight_decay,
        )
        muon_optimizer = torch.optim.Muon(
            muon_params, lr=self._learning_params.moun_matrix_lr, momentum=0.95
        )
        return [adamw_optimizer, muon_optimizer]

    def step(self, batch: dict[str, Any], phase: _Phase) -> None:

        if phase == _Phase.TRAINING and self._loss_aggregator is None:
            raise RuntimeError("Must have a loss aggregator object for training")

        y_init = self._model.core.y_init.repeat(
            (batch["tokens"].shape[0], batch["tokens"].shape[1], 1)
        ).to(batch["tokens"].device)
        z_init = self._model.core.z_init.repeat(
            batch["tokens"].shape[0], batch["tokens"], 1
        ).to(batch["tokens"].device)

        y = y_init
        z = z_init

        optimizers = self.optimizers()
        assert isinstance(
            optimizers, list
        ), "Something went wrong, there should be more than one optimizer"
        [opt.zero_grad() for opt in optimizers]

        total_loss_output = LossOutput(torch.zeros(1).to(batch["tokens"].device), {})

        rand_idx = None
        if self._learning_params.random_step_mask:
            rand_t = torch.rand_like(batch["tokens"].float())
            rand_idx = torch.argsort(rand_t, dim=-1)

        for idx in range(self._learning_params.supervision_steps):

            masked_token_input = batch["tokens"].clone()

            # Apply progressively random mask per step
            if self._learning_params.random_step_mask:

                mask = torch.zeros_like(batch["tokens"], dtype=torch.bool)
                assert (
                    rand_idx is not None
                ), "Makes the type checker happy; should never get there"
                sub_rand_idx = rand_idx[
                    :,
                    : int(
                        idx
                        / self._learning_params.supervision_steps
                        * batch["tokens"].shape[1]
                    ),
                ]
                for idx, sub_rand_idx_row in enumerate(sub_rand_idx):
                    mask[idx, sub_rand_idx_row] = True
                batch["mask"] = mask
            masked_token_input[~batch["mask"]] = self.model.core.vocab_size

            sup_step_output = self.forward(
                {"input": masked_token_input, "inter output": y, "latent": z}
            )
            sup_step_output["logits"] = sup_step_output["output"]
            y = sup_step_output["inter output"]
            z = sup_step_output["latent"]

            if self._loss_aggregator is None:
                continue

            loss = self._loss_aggregator(sup_step_output, batch)
            total_loss_output.total += (
                loss.total.detach() / self._learning_params.supervision_steps
            )
            for name in loss.individual:
                total_loss_output.individual[name] = (
                    total_loss_output.individual.get(name, 0)
                    + loss.individual[name].detach()
                    / self._learning_params.supervision_steps
                )

            if phase != _Phase.TRAINING:
                continue

            self.manual_backward(loss.total)

            # Apply gradient clip
            if self._learning_params.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self._learning_params.gradient_clip
                )

            # Apply optimizers
            [opt.step() for opt in optimizers]
            [opt.zero_grad() for opt in optimizers]

            # Break if q_stop is activated
            if torch.all(sup_step_output["stop"] > 0):
                break

        self.log_loss(total_loss_output, phase)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        self.step(batch, _Phase.TRAINING)

    def get_lr_multiplier(self, it: int) -> float:
        warmup_iters = round(
            self._learning_params.warmup_ratio * self._learning_params.num_iterations
        )
        warmdown_iters = round(self._learning_params.warmdown_ratio * num_iterations)
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        elif it <= num_iterations - warmdown_iters:
            return 1.0
        else:
            progress = (num_iterations - it) / warmdown_iters
            return (
                progress * 1.0 + (1 - progress) * self._learning_params.final_lr_ratio
            )
