"""Feature adapters for LG/ALG and the all-block Ours module."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class OfficialLGFeatureLoss(nn.Module):
    """Official LG stage projections, larger-grid resize, and summed mean MSE."""

    def __init__(
        self,
        student_feature_dims: Sequence[int],
        selected_student_indices: Sequence[int],
        teacher_channels: Sequence[int] = (16, 32, 64),
    ) -> None:
        super().__init__()
        self.selected_student_indices = tuple(int(v) for v in selected_student_indices)
        self.teacher_channels = tuple(int(v) for v in teacher_channels)
        if len(self.selected_student_indices) != len(self.teacher_channels):
            raise ValueError("LG requires one selected student feature per teacher stage")
        self.projections = nn.ModuleList(
            nn.Conv2d(
                int(student_feature_dims[index]),
                teacher_channel,
                kernel_size=1,
            )
            for index, teacher_channel in zip(
                self.selected_student_indices,
                self.teacher_channels,
                strict=True,
            )
        )

    def forward(
        self,
        student_features: Sequence[torch.Tensor],
        teacher_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(teacher_features) != len(self.teacher_channels):
            raise ValueError(
                f"Expected {len(self.teacher_channels)} teacher features, "
                f"got {len(teacher_features)}"
            )
        loss = teacher_features[0].new_zeros(())
        for index, projection, teacher_feature in zip(
            self.selected_student_indices,
            self.projections,
            teacher_features,
            strict=True,
        ):
            student_feature = projection(student_features[index])
            target_size = (
                max(student_feature.shape[-2], teacher_feature.shape[-2]),
                max(student_feature.shape[-1], teacher_feature.shape[-1]),
            )
            if student_feature.shape[-2:] != target_size:
                student_feature = F.interpolate(
                    student_feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            if teacher_feature.shape[-2:] != target_size:
                teacher_feature = F.interpolate(
                    teacher_feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            loss = loss + F.mse_loss(student_feature, teacher_feature)
        return loss


class OursAllBlockAdapter(nn.Module):
    """Convert heterogeneous Table-1 block grids to Ours' 192x14x14 contract.

    DeiT-Ti and ConViT-Ti already satisfy this contract and therefore use
    parameter-free identity paths.  Hierarchical and wider backbones receive
    an explicit per-block 1x1 projection and bilinear grid conversion before
    the unchanged all-block aggregation module.
    """

    def __init__(
        self,
        feature_dims: Sequence[int],
        *,
        output_channels: int = 192,
        output_grid: int = 14,
    ) -> None:
        super().__init__()
        self.output_channels = int(output_channels)
        self.output_grid = int(output_grid)
        self.projections = nn.ModuleList(
            nn.Identity()
            if int(channels) == self.output_channels
            else nn.Conv2d(int(channels), self.output_channels, kernel_size=1)
            for channels in feature_dims
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        if len(features) != len(self.projections):
            raise ValueError(
                f"Expected {len(self.projections)} block features, got {len(features)}"
            )
        converted: list[torch.Tensor] = []
        target_size = (self.output_grid, self.output_grid)
        for feature, projection in zip(features, self.projections, strict=True):
            output = projection(feature)
            if output.shape[-2:] != target_size:
                output = F.interpolate(
                    output,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            converted.append(output)
        return converted

