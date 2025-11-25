from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn


class LossComponent(ABC):
    """
    Basic loss component base class, using the __call__ method to compute the loss.

    Attributes:
        name (str): The name of the loss component.
        differentiable (bool): Whether the loss component is differentiable.
        weight (float): The weight of the loss component.
    """

    name: str
    differentiable: bool
    weight: float

    @abstractmethod
    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes the loss.

        Args:
            pred (dict[str, torch.Tensor]): A dictionary containing the predicted values.
            target (dict[str, torch.Tensor]): A dictionary containing the target values.

        Returns:
            torch.Tensor: The computed loss as a tensor.
        """
        ...


@dataclass
class BasicClassificationLoss(LossComponent):
    """
    Basic classification loss, most commonly cross entropy.

    Args:
        name (str): The name of the loss.
        weight (float): The weight of the loss.
        base_loss (nn.Module): The base loss function.
        differentiable (bool, optional): Whether the loss is differentiable. Defaults to True.

    Returns:
        torch.Tensor: The computed loss value.
    """

    name: str
    weight: float
    base_loss: nn.Module
    pred_key: str = "logits"
    ref_key: str = "class"
    differentiable: bool = True

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the loss.

        Args:
            pred (dict[str, torch.Tensor]): The predicted values.
            target (dict[str, torch.Tensor]): The target values.

        Returns:
            torch.Tensor: The computed loss value.
        """
        return self.base_loss(pred[self.pred_key], target[self.ref_key])


@dataclass
class BasicClassificationLossT(LossComponent):
    """
    Basic classification loss, most commonly cross entropy.

    Args:
        name (str): The name of the loss.
        weight (float): The weight of the loss.
        base_loss (nn.Module): The base loss function.
        differentiable (bool, optional): Whether the loss is differentiable. Defaults to True.

    Returns:
        torch.Tensor: The computed loss value.
    """

    name: str
    weight: float
    base_loss: nn.Module
    pred_key: str = "logits"
    ref_key: str = "class"
    differentiable: bool = True

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the loss.

        Args:
            pred (dict[str, torch.Tensor]): The predicted values.
            target (dict[str, torch.Tensor]): The target values.

        Returns:
            torch.Tensor: The computed loss value.
        """
        return self.base_loss(
            pred[self.pred_key].permute(0, 2, 1), target[self.ref_key]
        )


@dataclass
class HaltingCrossEntropy(LossComponent):
    """
    Cross entropy loss for computing the halting probability of the TRM.

    Attributes:
        name (str): The name of the loss component.
        weight (float): The weight of the loss component.
        base_loss (nn.Module): The base loss function.
        differentiable (bool, optional): Whether the loss is differentiable. Defaults to True.

    Returns:
        torch.Tensor: The computed loss
    """

    name: str
    weight: float
    base_loss: nn.Module
    differentiable: bool = True
    pred_stop_key: str = "stop"
    pred_logits_key: str = "logits"
    target_key: str = "class"
    mask_key: str = "mask"

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        prediction = pred[self.pred_stop_key]
        pred_logits = pred[self.pred_logits_key]

        mask = target[self.mask_key]
        target_halting_col = []
        for idx in range(pred_logits.shape[0]):
            target_halting_ind = torch.all(
                torch.argmax(pred_logits[idx][~mask[idx]], dim=1)
                == target[self.target_key][idx][~mask[idx]]
            )
            target_halting_col.append(target_halting_ind)
        target_halting = torch.stack(target_halting_col)

        return self.base_loss(
            prediction.squeeze(),
            target_halting.float().squeeze(),
        )


@dataclass
class PercentCorrect(LossComponent):
    """
    Percent correct metric for classification tasks.

    Args:
        name (str): The name of the metric.
        weight (float): The weight of the metric.
        differentiable (bool, optional): Whether the metric is differentiable. Defaults to False.

    Returns:
        torch.Tensor: The computed metric value.
    """

    name: str
    weight: float
    differentiable: bool = False

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the metric.

        Args:
            pred (dict[str, torch.Tensor]): The predicted values.
            target (dict[str, torch.Tensor]): The target values.

        Returns:
            torch.Tensor: The computed metric value.
        """
        pred_logits_argmax = torch.argmax(pred["logits"], dim=1)
        correct = torch.sum(pred_logits_argmax == target["class"])
        return correct / torch.numel(pred_logits_argmax)


@dataclass
class MaskedClassificationLoss(LossComponent):
    """
    Classification loss with masking, mostly used in Diffusion LLM training
    """

    name: str
    weight: float
    base_loss: nn.Module
    pred_key: str
    ref_key: str
    mask_key: str
    differentiable: bool = True

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Call method for outputting the loss

        Args:
            pred (dict[str, torch.Tensor]): Network estimation
            target (dict[str, torch.Tensor]): Ground truth reference

        Returns:
            torch.Tensor: loss
        """
        mask = ~target[self.mask_key]
        mask_size = mask.flatten().shape[0]
        if pred[self.pred_key].dim() == 4:
            logits = pred[self.pred_key][mask].permute(0, 2, 1)
        elif pred[self.pred_key].dim() == 3:
            logits = pred[self.pred_key][mask]
        labels = target[self.ref_key][mask]
        return self.base_loss(logits, labels) / mask_size  # type: ignore


@dataclass
class MaskedPercentCorrect(LossComponent):
    """
    Classification loss with masking, mostly used in Diffusion LLM training
    """

    name: str
    weight: float
    pred_key: str
    ref_key: str
    mask_key: str
    base_loss: nn.Module | None = None
    differentiable: bool = False

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Call method for outputting the loss

        Args:
            pred (dict[str, torch.Tensor]): Network estimation
            target (dict[str, torch.Tensor]): Ground truth reference

        Returns:
            torch.Tensor: loss
        """
        mask = ~target[self.mask_key]
        if pred[self.pred_key].dim() == 4:
            logits = pred[self.pred_key][mask].permute(0, 2, 1)
        elif pred[self.pred_key].dim() == 3:
            logits = pred[self.pred_key][mask]
        pred_logits_argmax = torch.argmax(logits, dim=1)  # type: ignore
        correct = torch.sum(pred_logits_argmax == target[self.ref_key][mask])
        return correct / torch.numel(pred_logits_argmax)
