# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PET (Point Edge Transformer) model wrapper.

Wraps the pure-torch internals of the PET architecture from the
`metatrain <https://github.com/metatensor/metatrain>`_ package as a
:class:`~nvalchemi.models.base.BaseModelMixin`-compatible wrapper. Unlike
:class:`~nvalchemi.models.mace.MACEWrapper` — which wraps an already
instantiated MACE module — :class:`PETWrapper` rebuilds a slim
:class:`_PETCore` that inherits only the pure-torch submodules from
``metatrain.pet.modules.*`` (``CartesianTransformer``, NEF helpers, cutoff
utilities). The metatomic-bound PET class is **not** reused, so the forward
path is ``torch.compile``-friendly and free of
``metatomic.torch.System`` / ``metatensor.torch.TensorMap`` dependencies
at call time.

Usage
-----
Load a PET checkpoint (e.g. ``pet-mad-xs-v1.5.0.ckpt``)::

    from nvalchemi.models.pet import PETWrapper
    import torch

    model = PETWrapper.from_checkpoint(
        "pet-mad-xs-v1.5.0.ckpt",
        device=torch.device("cuda"),
        dtype=torch.float32,
    )

Notes
-----
* Forces and stress are derived from the energy via autograd
  (``autograd_outputs = {"forces", "stress"}``). The non-conservative PET
  heads are intentionally skipped.
* The upstream composition model and scaler — originally wrapped as
  ``metatomic.torch.AtomisticModel`` with serialized ``TensorMap`` buffers
  — are decoded once at :meth:`PETWrapper.from_checkpoint` time into two
  flat torch buffers (``composition_energy``, ``scale_energy``) so the
  forward path has no metatensor dependency.
* Only the feedforward featurizer path is implemented (the residual path
  is not reachable from pet-mad-xs checkpoints).
* The long-range module is skipped entirely.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from nvalchemi._optional import OptionalDependency
from nvalchemi._typing import ModelOutputs
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models._utils import autograd_stresses, prepare_strain
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["PETWrapper"]


# ---------------------------------------------------------------------------
# Hyper-parameter normalisation
# ---------------------------------------------------------------------------


_REQUIRED_HYPERS: tuple[str, ...] = (
    "cutoff",
    "cutoff_function",
    "cutoff_width",
    "d_pet",
    "d_node",
    "d_head",
    "d_feedforward",
    "num_heads",
    "num_gnn_layers",
    "num_attention_layers",
    "normalization",
    "activation",
    "attention_temperature",
    "transformer_type",
    "featurizer_type",
)


def _validate_hypers(hypers: dict[str, Any]) -> None:
    """Raise :class:`ValueError` if *hypers* is missing a required key or uses
    an unsupported featurizer type.

    Parameters
    ----------
    hypers : dict[str, Any]
        Hyper-parameter dict pulled from the checkpoint's ``model_hypers``.

    Raises
    ------
    ValueError
        When a required key is missing or ``featurizer_type`` is not
        ``"feedforward"``.
    """
    missing = [key for key in _REQUIRED_HYPERS if key not in hypers]
    if missing:
        raise ValueError(f"PET hypers are missing required keys: {missing}")
    featurizer = hypers["featurizer_type"]
    if featurizer != "feedforward":
        raise ValueError(
            f"PETWrapper only supports the 'feedforward' featurizer "
            f"(got {featurizer!r}). Use metatrain.pet.PET directly if you "
            "need the residual path."
        )


# ---------------------------------------------------------------------------
# Core module
# ---------------------------------------------------------------------------


@OptionalDependency.PET.require
class _PETCore(nn.Module):
    """Pure-torch PET core — energy-head only, no metatomic/metatensor.

    Holds the exact ``nn.Module`` attributes that appear in a metatrain PET
    checkpoint state dict (minus the ``additive_models``, ``scaler``, long-range,
    and non-conservative heads we deliberately drop). The shape and naming
    conventions mirror :class:`metatrain.pet.PET` so a filtered state dict
    loads with ``strict=True``.

    Parameters
    ----------
    hypers : dict[str, Any]
        PET hyper-parameters. Must include the keys listed in
        :data:`_REQUIRED_HYPERS`. ``featurizer_type`` must be
        ``"feedforward"``.
    atomic_types : Sequence[int]
        Atomic numbers in the order used to build the species-index map.
        For pet-mad-xs this is ``[1, 2, ..., 102]``.

    Attributes
    ----------
    node_embedders : nn.ModuleList
        One :class:`torch.nn.Embedding` of shape ``[num_species, d_node]``
        (feedforward featurizer uses a single readout layer).
    edge_embedder : nn.Embedding
        Edge-species embedding of shape ``[num_species, d_pet]``.
    gnn_layers : nn.ModuleList
        ``num_gnn_layers`` x :class:`CartesianTransformer` blocks.
    combination_norms, combination_mlps : nn.ModuleList
        Per-layer bidirectional-message combiners.
    node_heads, edge_heads : nn.ModuleDict
        The per-output (``"energy"`` only) readout MLPs.
    node_last_layers, edge_last_layers : nn.ModuleDict
        The per-output final linear projections producing atomic energies.
    species_to_species_index : torch.Tensor
        Buffer of shape ``[max_Z + 1]`` mapping atomic numbers to their
        index in ``atomic_types``.
    """

    def __init__(
        self,
        hypers: dict[str, Any],
        atomic_types: Sequence[int],
    ) -> None:
        from metatrain.pet.modules.transformer import CartesianTransformer

        super().__init__()
        _validate_hypers(hypers)

        self.hypers = dict(hypers)
        self.atomic_types = list(atomic_types)
        self.cutoff = float(hypers["cutoff"])
        self.cutoff_function = str(hypers["cutoff_function"])
        self.cutoff_width = float(hypers["cutoff_width"])
        adaptive = hypers.get("num_neighbors_adaptive")
        self.num_neighbors_adaptive = float(adaptive) if adaptive is not None else None
        self.d_pet = int(hypers["d_pet"])
        self.d_node = int(hypers["d_node"])
        self.d_head = int(hypers["d_head"])
        self.num_readout_layers = 1  # feedforward featurizer

        num_species = len(self.atomic_types)
        self.gnn_layers = nn.ModuleList(
            [
                CartesianTransformer(
                    self.cutoff,
                    self.cutoff_width,
                    self.d_pet,
                    int(hypers["num_heads"]),
                    self.d_node,
                    int(hypers["d_feedforward"]),
                    int(hypers["num_attention_layers"]),
                    str(hypers["normalization"]),
                    str(hypers["activation"]),
                    float(hypers["attention_temperature"]),
                    str(hypers["transformer_type"]),
                    num_species,
                    layer_index == 0,
                )
                for layer_index in range(int(hypers["num_gnn_layers"]))
            ]
        )
        self.combination_norms = nn.ModuleList(
            [nn.LayerNorm(2 * self.d_pet) for _ in range(int(hypers["num_gnn_layers"]))]
        )
        self.combination_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * self.d_pet, 2 * self.d_pet),
                    nn.SiLU(),
                    nn.Linear(2 * self.d_pet, self.d_pet),
                )
                for _ in range(int(hypers["num_gnn_layers"]))
            ]
        )

        self.node_embedders = nn.ModuleList([nn.Embedding(num_species, self.d_node)])
        self.edge_embedder = nn.Embedding(num_species, self.d_pet)

        # Energy-only heads / last layers. Matching the metatrain layout
        # (ModuleDict["energy"] -> ModuleList[1] -> Sequential / ModuleDict)
        # is load-bearing for `load_state_dict(strict=True)`.
        self.node_heads = nn.ModuleDict(
            {
                "energy": nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(self.d_node, self.d_head),
                            nn.SiLU(),
                            nn.Linear(self.d_head, self.d_head),
                            nn.SiLU(),
                        )
                    ]
                )
            }
        )
        self.edge_heads = nn.ModuleDict(
            {
                "energy": nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(self.d_pet, self.d_head),
                            nn.SiLU(),
                            nn.Linear(self.d_head, self.d_head),
                            nn.SiLU(),
                        )
                    ]
                )
            }
        )
        self.node_last_layers = nn.ModuleDict(
            {
                "energy": nn.ModuleList(
                    [nn.ModuleDict({"energy___0": nn.Linear(self.d_head, 1)})]
                )
            }
        )
        self.edge_last_layers = nn.ModuleDict(
            {
                "energy": nn.ModuleList(
                    [nn.ModuleDict({"energy___0": nn.Linear(self.d_head, 1)})]
                )
            }
        )

        # Species-index map, matching the metatrain PET buffer shape.
        max_z = max(self.atomic_types) if self.atomic_types else 0
        species_to_species_index = torch.full((max_z + 1,), -1, dtype=torch.long)
        for i, z in enumerate(self.atomic_types):
            species_to_species_index[z] = i
        self.register_buffer(
            "species_to_species_index", species_to_species_index, persistent=True
        )

    # ------------------------------------------------------------------
    # Featurisation
    # ------------------------------------------------------------------

    def _featurize(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inlined feedforward featurization body.

        Mirrors :meth:`metatrain.pet.PET._feedforward_featurization_impl`,
        keeping only the tensors the downstream heads actually use.
        ``use_manual_attention`` is forced to ``False``; we only need
        inference (no double-backward).

        Parameters
        ----------
        inputs : dict[str, torch.Tensor]
            Dict populated by :meth:`PETWrapper.adapt_input`.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(node_features, edge_features)`` where ``node_features`` has
            shape ``[N, d_node]`` and ``edge_features`` has shape
            ``[N, max_edges_per_node, d_pet]``.
        """
        input_node_embeddings = self.node_embedders[0](inputs["element_indices_nodes"])
        input_edge_embeddings = self.edge_embedder(inputs["element_indices_neighbors"])
        reverse_idx = inputs["reverse_neighbor_index"]
        for combination_norm, combination_mlp, gnn_layer in zip(
            self.combination_norms,
            self.combination_mlps,
            self.gnn_layers,
            strict=True,
        ):
            output_node_embeddings, output_edge_embeddings = gnn_layer(
                input_node_embeddings,
                input_edge_embeddings,
                inputs["element_indices_neighbors"],
                inputs["edge_vectors"],
                inputs["padding_mask"],
                inputs["edge_distances"],
                inputs["cutoff_factors"],
                False,  # use_manual_attention: inference-only
            )
            input_node_embeddings = output_node_embeddings
            # Reverse the edge messages using the precomputed reverse index.
            flat = output_edge_embeddings.reshape(
                output_edge_embeddings.shape[0] * output_edge_embeddings.shape[1],
                output_edge_embeddings.shape[2],
            )
            new_input_edge_embeddings = flat[reverse_idx].reshape(
                output_edge_embeddings.shape
            )
            concatenated = torch.cat(
                [output_edge_embeddings, new_input_edge_embeddings], dim=-1
            )
            input_edge_embeddings = (
                input_edge_embeddings
                + output_edge_embeddings
                + combination_mlp(combination_norm(concatenated))
            )
        return input_node_embeddings, input_edge_embeddings

    def forward(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run featurisation and the energy head.

        Parameters
        ----------
        inputs : dict[str, torch.Tensor]
            Dict populated by :meth:`PETWrapper.adapt_input`.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(node_pred, edge_pred)`` per-atom energy contributions, each
            of shape ``[N, 1]``. The caller sums them and applies the
            scaler / composition buffers.
        """
        node_feats, edge_feats = self._featurize(inputs)
        node_head = self.node_heads["energy"][0]
        edge_head = self.edge_heads["energy"][0]
        node_last = self.node_last_layers["energy"][0]["energy___0"]
        edge_last = self.edge_last_layers["energy"][0]["energy___0"]

        node_ll = node_head(node_feats)  # [N, d_head]
        edge_ll = edge_head(edge_feats)  # [N, max_edges, d_head]

        node_pred = node_last(node_ll)  # [N, 1]
        # Apply edge last layer per-edge, then zero out padded slots, then
        # weight by cutoff factors and sum over neighbors. Applying edge_last
        # after the sum would miscompute the bias contribution.
        edge_per_edge = edge_last(edge_ll)  # [N, max_edges, 1]
        padding_mask = inputs["padding_mask"]  # [N, max_edges]
        edge_per_edge = torch.where(
            padding_mask.unsqueeze(-1), edge_per_edge, torch.zeros_like(edge_per_edge)
        )
        cutoff_factors = inputs["cutoff_factors"]  # [N, max_edges]
        edge_pred = (edge_per_edge * cutoff_factors.unsqueeze(-1)).sum(dim=1)  # [N, 1]
        return node_pred, edge_pred

    def compute_node_feats(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return node features only (for :meth:`PETWrapper.compute_embeddings`).

        Parameters
        ----------
        inputs : dict[str, torch.Tensor]
            Dict populated by :meth:`PETWrapper.adapt_input`.

        Returns
        -------
        torch.Tensor
            Node feature tensor of shape ``[N, d_node]``.
        """
        node_feats, _ = self._featurize(inputs)
        return node_feats


# ---------------------------------------------------------------------------
# State-dict filtering helpers
# ---------------------------------------------------------------------------


_DROP_PREFIXES: tuple[str, ...] = ("additive_models.", "scaler.")
_DROP_SUBSTRINGS: tuple[str, ...] = (
    ".non_conservative_forces.",
    ".non_conservative_stress.",
)


def _filter_state_dict(raw_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop checkpoint keys that belong to the metatomic-bound modules.

    Keeps everything :class:`_PETCore` needs, and discards the composition
    model, scaler, non-conservative heads, and the ``finetune_config``
    placeholder. The returned dict is ready for
    ``_PETCore.load_state_dict(strict=True)``.

    Parameters
    ----------
    raw_sd : dict[str, torch.Tensor]
        State dict from ``wrapped_model_checkpoint["model_state_dict"]``.

    Returns
    -------
    dict[str, torch.Tensor]
        Filtered state dict.
    """
    filtered: dict[str, torch.Tensor] = {}
    for key, value in raw_sd.items():
        if key == "finetune_config":
            continue
        if key.startswith(_DROP_PREFIXES):
            continue
        if any(sub in key for sub in _DROP_SUBSTRINGS):
            continue
        filtered[key] = value
    return filtered


def _decode_tensor_map_values(buffer: torch.Tensor) -> torch.Tensor:
    """Decode a serialized ``TensorMap`` buffer to a flat tensor.

    The metatrain composition / scaler modules store their weights as
    ``TensorMap`` byte buffers (uint8 tensors). Decoding with
    ``metatensor.torch.load_buffer`` returns a :class:`TensorMap`; the
    single block's ``values`` tensor carries the actual numeric weights.

    Parameters
    ----------
    buffer : torch.Tensor
        ``uint8`` tensor holding the serialized TensorMap.

    Returns
    -------
    torch.Tensor
        The decoded ``values`` tensor from the single block of the
        TensorMap.
    """
    import metatensor.torch as mts

    tensor_map = mts.load_buffer(buffer)
    return tensor_map.block(0).values


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


@OptionalDependency.PET.require
class PETWrapper(nn.Module, BaseModelMixin):
    """:class:`~nvalchemi.models.base.BaseModelMixin` wrapper around PET.

    Handles:

    * Building the input dict expected by :class:`_PETCore` from a
      :class:`~nvalchemi.data.Batch` — edge vectors, NEF reshaping,
      adaptive cutoffs.
    * Enabling gradients on ``positions`` when autograd outputs are active,
      and wiring the affine strain trick
      (:func:`~nvalchemi.models._utils.prepare_strain`) for stress.
    * Applying the flat composition / scaler buffers decoded from the
      checkpoint at load time.
    * Producing :class:`~nvalchemi._typing.ModelOutputs` with
      ``energy``, ``forces``, and ``stress``.

    Parameters
    ----------
    core : _PETCore
        Core module holding the pure-torch PET weights.
    atomic_types : Sequence[int]
        Atomic numbers in species-index order.
    hypers : dict[str, Any]
        PET hyper-parameters.
    composition_energy : torch.Tensor
        Per-species reference energy, shape ``[num_species]``. Indexed by
        the species index (not by atomic number).
    scale_energy : torch.Tensor
        Scalar (0-dim) tensor used as the global energy scale.

    Attributes
    ----------
    core : _PETCore
        Underlying PET core.
    atomic_types : list[int]
        Copy of the atomic-number list used to build the core.
    hypers : dict[str, Any]
        Copy of the hyper-parameters.
    model_config : ModelConfig
        Capability declaration with ``active_outputs`` defaulting to
        ``{"energy", "forces", "stress"}``.
    """

    core: _PETCore

    def __init__(
        self,
        core: _PETCore,
        atomic_types: Sequence[int],
        hypers: dict[str, Any],
        composition_energy: torch.Tensor,
        scale_energy: torch.Tensor,
    ) -> None:
        super().__init__()
        self.core = core
        self.atomic_types = list(atomic_types)
        self.hypers = dict(hypers)

        # Per-species reference energy (shape [num_species]) indexed by species
        # index (core.species_to_species_index lookup), not atomic number.
        # Non-persistent: decoded at `from_checkpoint` time from the raw
        # metatensor buffer; saved back via `export_model` using the same route.
        self.register_buffer(
            "composition_energy", composition_energy.clone(), persistent=False
        )
        # Scalar global scaler. Kept as a 0-dim tensor for broadcast.
        self.register_buffer(
            "scale_energy", scale_energy.clone().reshape(()), persistent=False
        )

        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress"}),
            autograd_outputs=frozenset({"forces", "stress"}),
            autograd_inputs=frozenset({"positions"}),
            required_inputs=frozenset(),
            optional_inputs=frozenset({"cell", "neighbor_list_shifts"}),
            supports_pbc=True,
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=float(hypers["cutoff"]),
                format=NeighborListFormat.COO,
                half_list=False,
            ),
        )

    # ------------------------------------------------------------------
    # BaseModelMixin required properties
    # ------------------------------------------------------------------

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return node/graph embedding shapes, both ``(d_node,)``."""
        return {
            "node_embeddings": (self.core.d_node,),
            "graph_embeddings": (self.core.d_node,),
        }

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def cutoff(self) -> float:
        """Interaction cutoff in Angstroms."""
        return self.core.cutoff

    @property
    def _model_dtype(self) -> torch.dtype:
        """Return the current dtype of the core's parameters.

        Read live from ``parameters()`` so it stays correct after
        ``.to(dtype=...)`` calls.
        """
        try:
            return next(self.core.parameters()).dtype
        except StopIteration:
            return torch.float32

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def _prepare_inputs(
        self, data: Batch, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        """Build the feature dict consumed by :class:`_PETCore`.

        Mirrors :func:`metatrain.pet.modules.structures.systems_to_batch`
        but sources every input from :class:`~nvalchemi.data.Batch` fields
        (``positions``, ``atomic_numbers``, ``neighbor_list``,
        ``neighbor_list_shifts``, ``cell``, ``batch_idx``) instead of
        ``metatomic.torch.System``.

        Parameters
        ----------
        data : Batch
            Input batch. ``positions`` is assumed to already be in *dtype*
            (with ``requires_grad`` set by the caller when appropriate).
        dtype : torch.dtype
            Model dtype; used to cast ``cell`` and ``neighbor_list_shifts``.

        Returns
        -------
        dict[str, torch.Tensor]
            Keys consumed by :meth:`_PETCore._featurize`:
            ``element_indices_nodes``, ``element_indices_neighbors``,
            ``edge_vectors``, ``edge_distances``, ``padding_mask``,
            ``reverse_neighbor_index``, ``cutoff_factors``.
        """
        from metatrain.pet.modules.adaptive_cutoff import get_adaptive_cutoffs
        from metatrain.pet.modules.nef import (
            compute_reversed_neighbor_list,
            edge_array_to_nef,
            get_corresponding_edges,
            get_nef_indices,
        )
        from metatrain.pet.modules.utilities import (
            cutoff_func_bump,
            cutoff_func_cosine,
        )

        positions = data.positions
        device = positions.device
        N = int(positions.shape[0])
        B = int(data.num_graphs)

        centers = data.neighbor_list[:, 0].long()
        neighbors = data.neighbor_list[:, 1].long()

        # Integer PBC shifts [E, 3] — zero for non-PBC systems.
        raw_shifts = getattr(data, "neighbor_list_shifts", None)
        if raw_shifts is None:
            cell_shifts = torch.zeros(
                centers.shape[0], 3, dtype=torch.long, device=device
            )
        else:
            cell_shifts = raw_shifts.to(dtype=torch.long, device=device)

        # Cell [B, 3, 3] — identity for non-PBC systems.
        raw_cell = getattr(data, "cell", None)
        if raw_cell is None:
            cell = (
                torch.eye(3, dtype=dtype, device=device)
                .unsqueeze(0)
                .expand(B, -1, -1)
                .contiguous()
            )
        else:
            cell = raw_cell.to(dtype=dtype, device=device)

        # Edge vectors with PBC contributions.
        cell_shifts_typed = cell_shifts.to(dtype=dtype)
        if B == 1:
            # Matches the upstream fast path; avoids the einsum when it's
            # not needed (slow backward for a single cell).
            cell_contributions = cell_shifts_typed @ cell[0]
        else:
            batch_idx_per_edge = data.batch_idx[centers].long()
            cell_contributions = torch.einsum(
                "ab,abc->ac",
                cell_shifts_typed,
                cell[batch_idx_per_edge],
            )
        edge_vectors = positions[neighbors] - positions[centers] + cell_contributions
        edge_distances = torch.norm(edge_vectors, dim=-1) + 1e-15

        # Adaptive cutoffs (optional).
        if self.core.num_neighbors_adaptive is not None:
            atomic_cutoffs = get_adaptive_cutoffs(
                centers,
                edge_distances,
                self.core.num_neighbors_adaptive,
                N,
                self.core.cutoff,
                cutoff_width=self.core.cutoff_width,
            )
            pair_cutoffs = (atomic_cutoffs[centers] + atomic_cutoffs[neighbors]) / 2.0
            cutoff_mask = edge_distances <= pair_cutoffs
            pair_cutoffs = pair_cutoffs[cutoff_mask]
            centers = centers[cutoff_mask]
            neighbors = neighbors[cutoff_mask]
            edge_vectors = edge_vectors[cutoff_mask]
            cell_shifts = cell_shifts[cutoff_mask]
            edge_distances = edge_distances[cutoff_mask]
        else:
            pair_cutoffs = self.core.cutoff * torch.ones(
                centers.shape[0], device=device, dtype=dtype
            )

        num_neighbors = torch.bincount(centers, minlength=N)
        max_edges_per_node = (
            int(torch.max(num_neighbors)) if num_neighbors.numel() > 0 else 0
        )

        cutoff_function = self.core.cutoff_function.lower()
        if cutoff_function == "bump":
            cutoff_factors = cutoff_func_bump(
                edge_distances, pair_cutoffs, self.core.cutoff_width
            )
        elif cutoff_function == "cosine":
            cutoff_factors = cutoff_func_cosine(
                edge_distances, pair_cutoffs, self.core.cutoff_width
            )
        else:
            raise ValueError(
                f"Unknown cutoff function type: {self.core.cutoff_function!r}. "
                "Supported types are 'Cosine' and 'Bump'."
            )

        # NEF reshaping.
        nef_indices, _, nef_mask = get_nef_indices(centers, N, max_edges_per_node)
        atomic_numbers = data.atomic_numbers.long()
        element_indices_nodes = self.core.species_to_species_index[atomic_numbers]
        element_indices_neighbors_flat = element_indices_nodes[neighbors]

        edge_vectors_nef = edge_array_to_nef(edge_vectors, nef_indices)
        edge_distances_nef = torch.sqrt(torch.sum(edge_vectors_nef**2, dim=2) + 1e-15)
        element_indices_neighbors = edge_array_to_nef(
            element_indices_neighbors_flat, nef_indices
        )
        cutoff_factors_nef = edge_array_to_nef(
            cutoff_factors, nef_indices, nef_mask, 0.0
        )

        corresponding_edges = get_corresponding_edges(
            torch.cat(
                [centers.unsqueeze(-1), neighbors.unsqueeze(-1), cell_shifts],
                dim=-1,
            )
        )
        reversed_neighbor_list = compute_reversed_neighbor_list(
            nef_indices, corresponding_edges, nef_mask
        )
        neighbors_index = edge_array_to_nef(neighbors, nef_indices).to(torch.int64)
        reverse_neighbor_index = (
            neighbors_index * neighbors_index.shape[1] + reversed_neighbor_list
        )
        # Replace padded indices with a unique sequence — fixes a backward
        # slowdown caused by duplicate gather indices (upstream comment
        # references pytorch#41162).
        num_padded = int(torch.sum(~nef_mask))
        if num_padded > 0:
            reverse_neighbor_index = reverse_neighbor_index.clone()
            reverse_neighbor_index[~nef_mask] = torch.arange(num_padded, device=device)

        return {
            "element_indices_nodes": element_indices_nodes,
            "element_indices_neighbors": element_indices_neighbors,
            "edge_vectors": edge_vectors_nef,
            "edge_distances": edge_distances_nef,
            "padding_mask": nef_mask,
            "reverse_neighbor_index": reverse_neighbor_index,
            "cutoff_factors": cutoff_factors_nef,
        }

    def adapt_input(
        self, data: AtomicData | Batch, **_kwargs: Any
    ) -> dict[str, torch.Tensor]:
        """Build the input dict expected by :class:`_PETCore`.

        Handles ``AtomicData -> Batch`` promotion and gradient enabling on
        ``positions`` when an autograd output is active. Strain handling
        (for stress) is done by :meth:`forward` **before** calling this
        method, so that the scaled positions/cell flow through the full
        featurisation.

        Parameters
        ----------
        data : AtomicData | Batch
            Input data. ``AtomicData`` inputs are promoted to a single-graph
            ``Batch``.
        **_kwargs : Any
            Ignored — kept for interface compatibility.

        Returns
        -------
        dict[str, torch.Tensor]
            Input dict for :class:`_PETCore`.
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        dtype = self._model_dtype

        # Cast positions to model dtype and enable gradients when needed.
        # Clone so the original batch tensor is never mutated in-place.
        positions = data.positions.to(dtype=dtype)
        if self.model_config.autograd_outputs & self.model_config.active_outputs:
            positions = positions.clone()
            positions.requires_grad_(True)
        # Store the prepared positions back on the batch so that downstream
        # autograd calls in `forward` can reach them via `data.positions`.
        data["positions"] = positions

        return self._prepare_inputs(data, dtype)

    def adapt_output(
        self,
        raw_output: dict[str, torch.Tensor | None],
        data: AtomicData | Batch,
    ) -> ModelOutputs:
        """Map raw PET outputs to the standard :class:`ModelOutputs` layout.

        Parameters
        ----------
        raw_output : dict[str, torch.Tensor | None]
            Dict with optional ``energy``, ``forces``, ``stress`` tensors.
        data : AtomicData | Batch
            Original input batch (unused here but forwarded to
            :meth:`BaseModelMixin.adapt_output`).

        Returns
        -------
        ModelOutputs
            Ordered dict keyed by the wrapper's active outputs.
        """
        mapped: dict[str, torch.Tensor] = {}
        energy = raw_output.get("energy")
        if energy is not None:
            mapped["energy"] = energy.unsqueeze(-1) if energy.ndim == 1 else energy
        if raw_output.get("forces") is not None:
            mapped["forces"] = raw_output["forces"]
        if raw_output.get("stress") is not None:
            mapped["stress"] = raw_output["stress"]
        return super().adapt_output(mapped, data)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        """Run the PET core and return energy / forces / stress.

        Conservative forces are derived via
        :func:`torch.autograd.grad` of the total energy with respect to
        positions. Stresses use the affine-strain trick from
        :func:`~nvalchemi.models._utils.prepare_strain` /
        :func:`~nvalchemi.models._utils.autograd_stresses` (same pattern as
        :class:`~nvalchemi.models.aimnet2.AIMNet2Wrapper`).

        Parameters
        ----------
        data : AtomicData | Batch
            Input batch.
        **kwargs : Any
            Forwarded to :meth:`adapt_input`.

        Returns
        -------
        ModelOutputs
            Dict with the active output keys populated.
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        active = self.model_config.active_outputs & self.model_config.outputs
        compute_forces = "forces" in active
        compute_stresses = "stress" in active

        # Set up the affine strain BEFORE adapt_input so the scaled positions
        # and cell flow through the full featurisation.
        displacement: torch.Tensor | None = None
        orig_positions: torch.Tensor | None = None
        orig_cell: torch.Tensor | None = None
        if compute_stresses and getattr(data, "cell", None) is not None:
            scaled_pos, scaled_cell, displacement = prepare_strain(
                data.positions.to(self._model_dtype),
                data.cell.to(self._model_dtype),
                data.batch_idx,
            )
            orig_positions = data.positions
            orig_cell = data.cell
            data["positions"] = scaled_pos
            data["cell"] = scaled_cell

        inputs = self.adapt_input(data, **kwargs)
        positions = data.positions  # updated in-place by adapt_input
        atomic_numbers = data.atomic_numbers.long()
        species_idx = self.core.species_to_species_index[atomic_numbers]

        node_pred, edge_pred = self.core(inputs)
        # Scaler first, then composition (matches upstream PET.forward order).
        per_atom = self.scale_energy * (node_pred + edge_pred)
        per_atom = per_atom + self.composition_energy[species_idx].unsqueeze(-1)

        B = int(data.num_graphs)
        energy = torch.zeros(B, 1, dtype=per_atom.dtype, device=per_atom.device)
        energy.scatter_add_(0, data.batch_idx.long().unsqueeze(-1), per_atom)

        result: dict[str, torch.Tensor] = {"energy": energy}

        if compute_forces:
            (grad,) = torch.autograd.grad(
                energy.sum(),
                positions,
                create_graph=False,
                retain_graph=compute_stresses,
            )
            result["forces"] = -grad

        if compute_stresses and displacement is not None and orig_cell is not None:
            result["stress"] = autograd_stresses(
                energy,
                displacement,
                orig_cell,
                B,
            )

        # Restore the batch's original positions/cell if strain was applied,
        # so the caller sees no mutation from the stress trick.
        if orig_positions is not None and orig_cell is not None:
            data["positions"] = orig_positions
            data["cell"] = orig_cell

        return self.adapt_output(result, data)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """Compute node and graph embeddings without autograd.

        Writes ``node_embeddings`` (``[N, d_node]``) and
        ``graph_embeddings`` (``[B, d_node]``, sum-pooled over atoms) into
        *data* and returns it. Does **not** mutate ``model_config``.

        Parameters
        ----------
        data : AtomicData | Batch
            Input data.
        **kwargs : Any
            Forwarded to :meth:`adapt_input`.

        Returns
        -------
        AtomicData | Batch
            The same batch with ``node_embeddings`` and
            ``graph_embeddings`` attached.
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        dtype = self._model_dtype

        with torch.no_grad():
            data["positions"] = data.positions.to(dtype=dtype)
            inputs = self._prepare_inputs(data, dtype)
            node_feats = self.core.compute_node_feats(inputs)

        # Write node embeddings directly to the atoms group to avoid the
        # default "system" routing used by `setattr` on unknown keys.
        atoms_group = data._atoms_group
        if atoms_group is not None:
            atoms_group["node_embeddings"] = node_feats
        else:
            data.node_embeddings = node_feats

        hidden_dim = node_feats.shape[-1]
        graph_embeddings = torch.zeros(
            data.num_graphs,
            hidden_dim,
            device=node_feats.device,
            dtype=node_feats.dtype,
        )
        graph_embeddings.scatter_add_(
            0,
            data.batch_idx.long().unsqueeze(-1).expand(-1, hidden_dim),
            node_feats,
        )
        data.graph_embeddings = graph_embeddings
        return data

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype | None = None,
    ) -> "PETWrapper":
        """Load a PET checkpoint from disk and return a wrapped instance.

        Expects a ``torch.load``-friendly dict written by
        :class:`metatrain.pet.PET`. The outer dict is an LLPR wrapper; the
        PET core lives in ``wrapped_model_checkpoint`` if that key is
        present, otherwise directly at the top level.

        Metatomic's torch extension **must** be importable before
        :func:`torch.load` is called, otherwise the checkpoint's
        ``ScriptObject`` metadata fails to unpickle. We import
        :mod:`metatomic.torch` lazily here to enforce that without
        polluting module import time.

        Parameters
        ----------
        checkpoint_path : Path | str
            Path to a PET checkpoint file (``.ckpt`` / ``.pt``).
        device : torch.device, optional
            Target device. Defaults to CPU.
        dtype : torch.dtype | None, optional
            If set, cast the core and composition/scaler buffers to this
            dtype before returning.

        Returns
        -------
        PETWrapper

        Raises
        ------
        OptionalDependencyError
            When :mod:`metatrain` is not installed.
        """
        if not OptionalDependency.PET.is_available():
            OptionalDependency.PET._raise_error(f"{cls.__qualname__}.from_checkpoint")
        # Make sure metatomic's custom torch ops are registered before
        # torch.load, otherwise the ScriptObject metadata unpickling fails.
        import metatomic.torch  # noqa: F401

        raw = torch.load(str(checkpoint_path), weights_only=False, map_location=device)
        if isinstance(raw, dict) and "wrapped_model_checkpoint" in raw:
            raw = raw["wrapped_model_checkpoint"]

        model_data = raw["model_data"]
        hypers = dict(model_data["model_hypers"])
        atomic_types = list(model_data["dataset_info"].atomic_types)
        raw_sd = raw["model_state_dict"]

        composition_values = _decode_tensor_map_values(
            raw_sd["additive_models.0.energy_composition_buffer"]
        )  # [num_species, 1]
        composition_energy = composition_values.squeeze(-1).clone()
        scale_values = _decode_tensor_map_values(
            raw_sd["scaler.energy_scaler_buffer"]
        )  # [1, 1]
        scale_energy = scale_values.reshape(()).clone()

        core_sd = _filter_state_dict(raw_sd)
        core = _PETCore(hypers, atomic_types)
        core.load_state_dict(core_sd, strict=True)

        if dtype is not None:
            core = core.to(dtype=dtype)
            composition_energy = composition_energy.to(dtype=dtype)
            scale_energy = scale_energy.to(dtype=dtype)

        wrapper = cls(
            core=core,
            atomic_types=atomic_types,
            hypers=hypers,
            composition_energy=composition_energy,
            scale_energy=scale_energy,
        )
        return wrapper.to(device)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        """Serialize the wrapper to disk in a pure-torch layout.

        Writes a plain dict containing the core ``state_dict``, the
        hyper-parameters, the atomic-type list, and the
        composition/scaler buffers. The output is **not** a metatrain /
        metatomic checkpoint — it is a self-contained snapshot that can
        be reloaded by constructing ``_PETCore(hypers, atomic_types)`` and
        calling ``load_state_dict`` on the saved dict.

        Parameters
        ----------
        path : Path
            Output path.
        as_state_dict : bool, optional
            If ``True``, save only the core's ``state_dict``. Defaults to
            ``False`` (saves the full snapshot).
        """
        if as_state_dict:
            torch.save(self.core.state_dict(), path)
        else:
            snapshot = {
                "core_state_dict": self.core.state_dict(),
                "hypers": self.hypers,
                "atomic_types": self.atomic_types,
                "composition_energy": self.composition_energy.detach().cpu(),
                "scale_energy": self.scale_energy.detach().cpu(),
            }
            torch.save(snapshot, path)
