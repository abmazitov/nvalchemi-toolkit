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
"""Tests for :class:`~nvalchemi.models.pet.PETWrapper`.

All tests in this module require :mod:`metatrain` (and the ``metatomic`` /
``metatensor`` packages that ride along with it) and are automatically
skipped when they are not installed. Install with::

    pip install 'nvalchemi-toolkit[pet]'
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

# metatrain.pet.__init__ imports metatrain.pet.trainer, which imports
# metatrain.utils.distributed.slurm, which pulls in `hostlist`. That package
# is a SLURM-only helper that the nvalchemi dev environment explicitly
# disables via the `override-dependencies` block in `pyproject.toml`. When
# it isn't present, stub a minimal module so the import chain resolves and
# the tests can still exercise PET's pure-torch code paths.
if "hostlist" not in sys.modules:
    try:
        import hostlist  # noqa: F401
    except ImportError:
        sys.modules["hostlist"] = types.ModuleType("hostlist")

# Skip the entire module when metatrain is not installed.
pytest.importorskip("metatrain", reason="metatrain not installed; skipping PET tests")

from nvalchemi.data import AtomicData, Batch  # noqa: E402
from nvalchemi.models.base import NeighborListFormat  # noqa: E402
from nvalchemi.models.pet import PETWrapper, _PETCore  # noqa: E402
from nvalchemi.neighbors import compute_neighbors

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_ATOMIC_NUMBERS = [1, 6, 8]  # H, C, O — keeps species_to_species_index small
_CUTOFF = 5.0
_D_NODE = 16
_D_PET = 8


def _tiny_hypers() -> dict:
    """Return a minimal PET hyper-parameter dict for fast unit tests."""
    return {
        "cutoff": _CUTOFF,
        "cutoff_width": 0.5,
        "cutoff_function": "Bump",
        "d_pet": _D_PET,
        "d_node": _D_NODE,
        "d_head": 8,
        "d_feedforward": 8,
        "num_heads": 2,
        "num_gnn_layers": 1,
        "num_attention_layers": 1,
        "normalization": "RMSNorm",
        "activation": "SwiGLU",
        "attention_temperature": 1.0,
        "transformer_type": "PreLN",
        "featurizer_type": "feedforward",
        "num_neighbors_adaptive": None,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_water(device: str = "cpu") -> AtomicData:
    """Single H2O molecule with a pre-computed full edge list (no PBC)."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    numbers = torch.tensor([8, 1, 1], dtype=torch.long, device=device)
    neighbor_list = torch.tensor(
        [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]],
        dtype=torch.long,
        device=device,
    )
    neighbor_list_shifts = torch.zeros(
        neighbor_list.shape[0], 3, dtype=torch.long, device=device
    )
    return AtomicData(
        positions=positions,
        atomic_numbers=numbers,
        neighbor_list=neighbor_list,
        neighbor_list_shifts=neighbor_list_shifts,
    )


def _make_two_atoms(device: str = "cpu") -> AtomicData:
    """Two H atoms at (0, 0, 0) and (1.1, 0, 0) with a symmetric edge pair."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=torch.float32, device=device
    )
    numbers = torch.tensor([1, 1], dtype=torch.long, device=device)
    neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    neighbor_list_shifts = torch.zeros(2, 3, dtype=torch.long, device=device)
    return AtomicData(
        positions=positions,
        atomic_numbers=numbers,
        neighbor_list=neighbor_list,
        neighbor_list_shifts=neighbor_list_shifts,
    )


def _make_pbc_water(device: str = "cpu") -> AtomicData:
    """H2O in a 10 Å cubic periodic box."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    numbers = torch.tensor([8, 1, 1], dtype=torch.long, device=device)
    neighbor_list = torch.tensor(
        [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]],
        dtype=torch.long,
        device=device,
    )
    cell = (torch.eye(3, dtype=torch.float32, device=device) * 10.0).unsqueeze(0)
    neighbor_list_shifts = torch.zeros(6, 3, dtype=torch.long, device=device)
    pbc = torch.tensor([[True, True, True]], device=device)
    return AtomicData(
        positions=positions,
        atomic_numbers=numbers,
        neighbor_list=neighbor_list,
        cell=cell,
        neighbor_list_shifts=neighbor_list_shifts,
        pbc=pbc,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_core() -> _PETCore:
    """Small real PET core — cheap enough for the fast test path."""
    torch.manual_seed(0)
    return _PETCore(_tiny_hypers(), _ATOMIC_NUMBERS)


@pytest.fixture
def wrapper(tiny_core) -> PETWrapper:
    """PETWrapper with zero composition and unit scaler."""
    composition = torch.zeros(len(_ATOMIC_NUMBERS), dtype=torch.float32)
    scale = torch.tensor(1.0, dtype=torch.float32)
    return PETWrapper(
        core=tiny_core,
        atomic_types=_ATOMIC_NUMBERS,
        hypers=_tiny_hypers(),
        composition_energy=composition,
        scale_energy=scale,
    )


@pytest.fixture
def single_batch() -> Batch:
    return Batch.from_data_list([_make_water()])


@pytest.fixture
def multi_batch() -> Batch:
    """Two H2O molecules as a batched system (B=2, N=6)."""
    return Batch.from_data_list([_make_water(), _make_water()])


@pytest.fixture
def pbc_batch() -> Batch:
    return Batch.from_data_list([_make_pbc_water()])


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_wrapper_holds_core(self, tiny_core):
        composition = torch.zeros(len(_ATOMIC_NUMBERS))
        scale = torch.tensor(1.0)
        w = PETWrapper(
            core=tiny_core,
            atomic_types=_ATOMIC_NUMBERS,
            hypers=_tiny_hypers(),
            composition_energy=composition,
            scale_energy=scale,
        )
        assert w.core is tiny_core

    def test_default_model_config(self, wrapper):
        assert "forces" in wrapper.model_config.active_outputs
        assert "stress" in wrapper.model_config.active_outputs

    def test_composition_buffer_shape(self, wrapper):
        assert wrapper.composition_energy.shape == (len(_ATOMIC_NUMBERS),)

    def test_scale_buffer_scalar(self, wrapper):
        assert wrapper.scale_energy.shape == ()

    def test_buffers_not_in_state_dict(self, wrapper):
        # composition_energy / scale_energy are non-persistent.
        sd_keys = wrapper.state_dict().keys()
        assert not any(k.endswith("composition_energy") for k in sd_keys)
        assert not any(k.endswith("scale_energy") for k in sd_keys)

    def test_validate_hypers_rejects_missing(self):
        bad = _tiny_hypers()
        del bad["cutoff"]
        with pytest.raises(ValueError, match="missing required keys"):
            _PETCore(bad, _ATOMIC_NUMBERS)

    def test_validate_hypers_rejects_non_feedforward(self):
        bad = _tiny_hypers()
        bad["featurizer_type"] = "residual"
        with pytest.raises(ValueError, match="feedforward"):
            _PETCore(bad, _ATOMIC_NUMBERS)

    def test_import_error_without_metatrain(self, monkeypatch, tiny_core):
        from nvalchemi._optional import OptionalDependency

        monkeypatch.setattr(OptionalDependency.PET, "_available", False)
        with pytest.raises(ImportError):
            PETWrapper(
                core=tiny_core,
                atomic_types=_ATOMIC_NUMBERS,
                hypers=_tiny_hypers(),
                composition_energy=torch.zeros(len(_ATOMIC_NUMBERS)),
                scale_energy=torch.tensor(1.0),
            )


# ---------------------------------------------------------------------------
# ModelConfig capability checks
# ---------------------------------------------------------------------------


class TestModelConfigCapabilities:
    def test_forces_via_autograd(self, wrapper):
        assert "forces" in wrapper.model_config.autograd_outputs

    def test_stress_via_autograd(self, wrapper):
        assert "stress" in wrapper.model_config.autograd_outputs

    def test_outputs_include_energies_forces_stresses(self, wrapper):
        cfg = wrapper.model_config
        assert "energy" in cfg.outputs
        assert "forces" in cfg.outputs
        assert "stress" in cfg.outputs

    def test_autograd_inputs(self, wrapper):
        assert "positions" in wrapper.model_config.autograd_inputs

    def test_supports_pbc(self, wrapper):
        assert wrapper.model_config.supports_pbc is True

    def test_embedding_shapes_available(self, wrapper):
        shapes = wrapper.embedding_shapes
        assert "node_embeddings" in shapes
        assert "graph_embeddings" in shapes

    def test_neighbor_config_coo(self, wrapper):
        nc = wrapper.model_config.neighbor_config
        assert nc is not None
        assert nc.format == NeighborListFormat.COO
        assert nc.cutoff == pytest.approx(_CUTOFF)
        assert nc.half_list is False

    def test_needs_pbc_false(self, wrapper):
        assert wrapper.model_config.needs_pbc is False


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_cutoff(self, wrapper):
        assert wrapper.cutoff == pytest.approx(_CUTOFF)
        assert isinstance(wrapper.cutoff, float)

    def test_embedding_shapes(self, wrapper):
        shapes = wrapper.embedding_shapes
        assert shapes["node_embeddings"] == (_D_NODE,)
        assert shapes["graph_embeddings"] == (_D_NODE,)

    def test_model_dtype(self, wrapper):
        assert wrapper._model_dtype == torch.float32


# ---------------------------------------------------------------------------
# adapt_input
# ---------------------------------------------------------------------------


class TestAdaptInput:
    def test_required_keys_present(self, wrapper, single_batch):
        inp = wrapper.adapt_input(single_batch)
        for key in (
            "element_indices_nodes",
            "element_indices_neighbors",
            "edge_vectors",
            "edge_distances",
            "padding_mask",
            "reverse_neighbor_index",
            "cutoff_factors",
        ):
            assert key in inp, f"Missing key: {key}"

    def test_element_indices_nodes(self, wrapper, single_batch):
        # atomic_numbers for H2O = [8, 1, 1]; _ATOMIC_NUMBERS = [1, 6, 8]
        # → species indices [2, 0, 0]
        inp = wrapper.adapt_input(single_batch)
        assert inp["element_indices_nodes"].tolist() == [2, 0, 0]

    def test_edge_vector_shape(self, wrapper, single_batch):
        inp = wrapper.adapt_input(single_batch)
        # NEF layout: [N, max_edges_per_node, 3]
        assert inp["edge_vectors"].shape[0] == 3  # N atoms
        assert inp["edge_vectors"].shape[-1] == 3  # xyz

    def test_positions_requires_grad_when_forces_active(self, wrapper, single_batch):
        wrapper.model_config.active_outputs = {"energy", "forces"}
        wrapper.adapt_input(single_batch)
        assert single_batch.positions.requires_grad

    def test_positions_no_grad_energy_only(self, wrapper, single_batch):
        wrapper.model_config.active_outputs = {"energy"}
        wrapper.adapt_input(single_batch)
        assert not single_batch.positions.requires_grad

    def test_atomic_data_promoted_to_batch(self, wrapper):
        data = _make_water()
        inp = wrapper.adapt_input(data)
        assert inp["element_indices_nodes"].shape == (3,)

    def test_multi_batch_shapes(self, wrapper, multi_batch):
        inp = wrapper.adapt_input(multi_batch)
        # 6 atoms total across 2 water molecules.
        assert inp["element_indices_nodes"].shape == (6,)

    def test_pbc_runs(self, wrapper, pbc_batch):
        inp = wrapper.adapt_input(pbc_batch)
        assert inp["edge_vectors"].shape[0] == 3


# ---------------------------------------------------------------------------
# adapt_output
# ---------------------------------------------------------------------------


class TestAdaptOutput:
    def test_energy_key_in_output(self, wrapper, single_batch):
        raw = {"energy": torch.randn(1, 1)}
        out = wrapper.adapt_output(raw, single_batch)
        assert "energy" in out

    def test_energies_shape(self, wrapper, single_batch):
        raw = {"energy": torch.randn(1, 1)}
        out = wrapper.adapt_output(raw, single_batch)
        assert out["energy"].shape == (1, 1)

    def test_1d_energy_unsqueezed(self, wrapper, single_batch):
        raw = {"energy": torch.randn(1)}
        out = wrapper.adapt_output(raw, single_batch)
        assert out["energy"].shape == (1, 1)

    def test_forces_passed_through(self, wrapper, single_batch):
        wrapper.model_config.active_outputs = {"energy", "forces"}
        raw = {"energy": torch.randn(1, 1), "forces": torch.randn(3, 3)}
        out = wrapper.adapt_output(raw, single_batch)
        assert out["forces"].shape == (3, 3)

    def test_stress_passed_through(self, wrapper, single_batch):
        wrapper.model_config.active_outputs = {"energy", "forces", "stress"}
        raw = {
            "energy": torch.randn(1, 1),
            "forces": torch.randn(3, 3),
            "stress": torch.randn(1, 3, 3),
        }
        out = wrapper.adapt_output(raw, single_batch)
        assert out["stress"].shape == (1, 3, 3)


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------


class TestForward:
    def test_energies_shape_single(self, wrapper, single_batch):
        out = wrapper.forward(single_batch)
        assert out["energy"].shape == (1, 1)

    def test_energies_shape_multi(self, wrapper, multi_batch):
        out = wrapper.forward(multi_batch)
        assert out["energy"].shape == (2, 1)

    def test_energies_dtype(self, wrapper, single_batch):
        out = wrapper.forward(single_batch)
        assert out["energy"].dtype == wrapper._model_dtype

    def test_forces_shape(self, wrapper, single_batch):
        out = wrapper.forward(single_batch)
        assert out["forces"].shape == (3, 3)

    def test_forces_shape_multi(self, wrapper, multi_batch):
        out = wrapper.forward(multi_batch)
        assert out["forces"].shape == (6, 3)

    def test_no_forces_when_disabled(self, wrapper, single_batch):
        wrapper.model_config.active_outputs = {"energy"}
        out = wrapper.forward(single_batch)
        assert out.get("forces") is None

    def test_atomic_data_input(self, wrapper):
        data = _make_water()
        out = wrapper.forward(data)
        assert out["energy"].shape == (1, 1)

    def test_pbc_stress_shape(self, wrapper, pbc_batch):
        out = wrapper.forward(pbc_batch)
        assert out["stress"].shape == (1, 3, 3)

    def test_forces_match_finite_difference(self, wrapper):
        """Conservative forces agree with a numerical gradient.

        Uses a small two-atom system so the finite-difference evaluation is
        cheap, and a ``float64`` copy of the wrapper so the FD comparison
        isn't dominated by float32 rounding.
        """
        core = _PETCore(_tiny_hypers(), _ATOMIC_NUMBERS).to(torch.float64)
        w = PETWrapper(
            core=core,
            atomic_types=_ATOMIC_NUMBERS,
            hypers=_tiny_hypers(),
            composition_energy=torch.zeros(len(_ATOMIC_NUMBERS), dtype=torch.float64),
            scale_energy=torch.tensor(1.0, dtype=torch.float64),
        )
        data = _make_two_atoms()
        data["positions"] = data.positions.to(torch.float64)
        batch = Batch.from_data_list([data])
        out = w.forward(batch)
        analytic_force = out["forces"][0, 0].item()

        eps = 1e-4
        base_pos = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=torch.float64)

        def energy_at(pos):
            d = AtomicData(
                positions=pos,
                atomic_numbers=torch.tensor([1, 1], dtype=torch.long),
                neighbor_list=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                neighbor_list_shifts=torch.zeros(2, 3, dtype=torch.long),
            )
            b = Batch.from_data_list([d])
            w.model_config.active_outputs = {"energy"}
            return w.forward(b)["energy"].item()

        # Restore autograd-output expectation after the energy-only sanity calls.
        w.model_config.active_outputs = {"energy", "forces", "stress"}

        pos_p = base_pos.clone()
        pos_p[0, 0] += eps
        pos_m = base_pos.clone()
        pos_m[0, 0] -= eps
        fd = -(energy_at(pos_p) - energy_at(pos_m)) / (2 * eps)

        assert analytic_force == pytest.approx(fd, abs=1e-3)


# ---------------------------------------------------------------------------
# compute_embeddings
# ---------------------------------------------------------------------------


class TestComputeEmbeddings:
    def test_node_embeddings_shape(self, wrapper, single_batch):
        result = wrapper.compute_embeddings(single_batch)
        assert result.node_embeddings.shape == (3, _D_NODE)

    def test_graph_embeddings_shape(self, wrapper, single_batch):
        result = wrapper.compute_embeddings(single_batch)
        assert result.graph_embeddings.shape == (1, _D_NODE)

    def test_graph_embeddings_shape_multi(self, wrapper, multi_batch):
        result = wrapper.compute_embeddings(multi_batch)
        assert result.graph_embeddings.shape == (2, _D_NODE)

    def test_graph_embeddings_is_sum_of_node_embeddings(self, wrapper, single_batch):
        result = wrapper.compute_embeddings(single_batch)
        expected_graph = result.node_embeddings.sum(dim=0)
        assert torch.allclose(result.graph_embeddings[0], expected_graph)

    def test_does_not_mutate_model_config(self, wrapper, single_batch):
        wrapper.model_config.active_outputs = {"energy", "forces", "stress"}
        wrapper.compute_embeddings(single_batch)
        assert "forces" in wrapper.model_config.active_outputs
        assert "stress" in wrapper.model_config.active_outputs

    def test_atomic_data_input(self, wrapper):
        data = _make_water()
        result = wrapper.compute_embeddings(data)
        assert result.node_embeddings.shape == (3, _D_NODE)

    def test_no_grad_on_positions_after_embeddings(self, wrapper, single_batch):
        wrapper.compute_embeddings(single_batch)
        assert not single_batch.positions.requires_grad


# ---------------------------------------------------------------------------
# export_model
# ---------------------------------------------------------------------------


class TestExportModel:
    def test_export_snapshot(self, wrapper, tmp_path):
        path = tmp_path / "pet.pt"
        wrapper.export_model(path)
        assert path.exists()
        loaded = torch.load(path, weights_only=False)
        assert isinstance(loaded, dict)
        assert "core_state_dict" in loaded
        assert "hypers" in loaded
        assert "atomic_types" in loaded
        assert "composition_energy" in loaded
        assert "scale_energy" in loaded

    def test_export_state_dict(self, wrapper, tmp_path):
        path = tmp_path / "pet_sd.pt"
        wrapper.export_model(path, as_state_dict=True)
        assert path.exists()
        sd = torch.load(path, weights_only=True)
        assert isinstance(sd, dict)
        assert any("gnn_layers" in k for k in sd.keys())

    def test_reload_snapshot_into_new_core(self, wrapper, tmp_path):
        path = tmp_path / "pet.pt"
        wrapper.export_model(path)
        snapshot = torch.load(path, weights_only=False)

        new_core = _PETCore(snapshot["hypers"], snapshot["atomic_types"])
        new_core.load_state_dict(snapshot["core_state_dict"], strict=True)
        # Check a random parameter matches.
        for key in wrapper.core.state_dict():
            assert torch.allclose(
                new_core.state_dict()[key], wrapper.core.state_dict()[key]
            )


# ---------------------------------------------------------------------------
# from_checkpoint error path
# ---------------------------------------------------------------------------


class TestFromCheckpointErrors:
    def test_raises_import_error_when_metatrain_unavailable(self, monkeypatch):
        from nvalchemi._optional import OptionalDependency

        monkeypatch.setattr(OptionalDependency.PET, "_available", False)
        with pytest.raises(ImportError):
            PETWrapper.from_checkpoint("nonexistent.ckpt")


# ---------------------------------------------------------------------------
# Integration tests — real PET checkpoint (marked slow)
# ---------------------------------------------------------------------------


_CHECKPOINT_PATH = "pet-mad-xs-v1.5.0.ckpt"
_CUTOFF = 7.5


def _crystal(device: str = "cpu") -> AtomicData:
    return AtomicData(
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=torch.float32
        ).to(device=device),
        atomic_numbers=torch.tensor([6, 6], dtype=torch.long).to(device=device),
        cell=torch.eye(3, dtype=torch.float32).reshape(1, 3, 3).to(device=device),
        pbc=torch.tensor([True] * 3, dtype=torch.bool).reshape(1, 3).to(device=device),
    )


def _batch(dtype: torch.dtype = torch.float32) -> Batch:
    data = _crystal()
    batch = Batch.from_data_list([data])
    compute_neighbors(batch, cutoff=_CUTOFF, format=NeighborListFormat.COO)
    return batch


@pytest.fixture(scope="session")
def real_wrapper_cpu():
    """Load the pet-mad-xs checkpoint once per session."""
    import os

    if not os.path.exists(_CHECKPOINT_PATH):
        pytest.skip(
            f"Checkpoint {_CHECKPOINT_PATH} not found — "
            "download from HuggingFace to enable the slow tests."
        )
    try:
        return PETWrapper.from_checkpoint(
            _CHECKPOINT_PATH, device=torch.device("cpu"), dtype=torch.float32
        )
    except Exception as e:
        pytest.skip(f"Could not load PET checkpoint: {e}")


@pytest.mark.slow
class TestRealCheckpoint:
    """Integration tests against the pet-mad-xs-v1.5.0 checkpoint."""

    def test_is_pet_wrapper(self, real_wrapper_cpu):
        assert isinstance(real_wrapper_cpu, PETWrapper)

    def test_model_config_matches_wrapper(self, real_wrapper_cpu):
        cfg = real_wrapper_cpu.model_config
        assert "forces" in cfg.autograd_outputs
        assert "stress" in cfg.autograd_outputs
        assert "energy" in cfg.outputs
        assert cfg.neighbor_config is not None
        assert cfg.neighbor_config.format == NeighborListFormat.COO

    def test_cutoff_positive(self, real_wrapper_cpu):
        assert real_wrapper_cpu.cutoff > 0.0

    def test_inference_energy_finite(self, real_wrapper_cpu):
        batch = _batch()
        out = real_wrapper_cpu.forward(batch)
        e = out["energy"]
        assert e.shape == (1, 1)
        assert torch.isfinite(e).all()

    def test_inference_forces_shape(self, real_wrapper_cpu):
        batch = _batch()
        out = real_wrapper_cpu.forward(batch)
        assert out["forces"].shape == (2, 3)
        assert torch.isfinite(out["forces"]).all()

    def test_inference_stress_shape(self, real_wrapper_cpu):
        """Non-PBC system produces a stress tensor of zeros with shape [B, 3, 3]."""
        batch = _batch()
        out = real_wrapper_cpu.forward(batch)
        assert out["stress"].shape == (1, 3, 3)
        assert torch.isfinite(out["stress"]).all()

    def test_batch_determinism(self, real_wrapper_cpu):
        """Same input → same output across two consecutive calls."""
        b1 = _batch()
        b2 = _batch()
        e1 = real_wrapper_cpu.forward(b1)["energy"]
        e2 = real_wrapper_cpu.forward(b2)["energy"]
        assert torch.allclose(e1, e2, atol=1e-6)

    def test_batched_matches_single(self, real_wrapper_cpu):
        """Two-water batch energies equal the single-water energy."""
        single = real_wrapper_cpu.forward(_batch())
        multi_batch = Batch.from_data_list([_crystal(), _crystal()])
        compute_neighbors(multi_batch, cutoff=_CUTOFF, format=NeighborListFormat.COO)
        multi = real_wrapper_cpu.forward(multi_batch)
        assert torch.allclose(
            multi["energy"][0], single["energy"][0], atol=1e-4, rtol=1e-4
        )
        assert torch.allclose(
            multi["energy"][1], single["energy"][0], atol=1e-4, rtol=1e-4
        )

    def test_compute_embeddings_run(self, real_wrapper_cpu):
        batch = _batch()
        result = real_wrapper_cpu.compute_embeddings(batch)
        assert result.node_embeddings.shape[0] == 2
        assert result.graph_embeddings.shape == (1, result.node_embeddings.shape[1])

    def test_export_and_reload(self, real_wrapper_cpu, tmp_path):
        path = tmp_path / "pet_snapshot.pt"
        real_wrapper_cpu.export_model(path)
        snapshot = torch.load(path, weights_only=False)
        new_core = _PETCore(snapshot["hypers"], snapshot["atomic_types"])
        new_core.load_state_dict(snapshot["core_state_dict"], strict=True)

    def test_metatrain_model_compatibility(self, real_wrapper_cpu):
        """Predictions from the PETWrapper match those from the original metatrain model."""
        ase = pytest.importorskip("ase")
        from metatomic.torch import NeighborListOptions, systems_to_torch
        from metatrain.utils.data.target_info import get_energy_target_info
        from metatrain.utils.evaluate_model import evaluate_model
        from metatrain.utils.io import load_model
        from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

        mt_model = load_model(_CHECKPOINT_PATH)
        data = _crystal()
        atoms = ase.Atoms(
            positions=data.positions.cpu().numpy(),
            numbers=data.atomic_numbers.cpu().numpy(),
            cell=data.cell.cpu().squeeze().numpy() if data.cell is not None else None,
            pbc=data.pbc.cpu().squeeze().numpy() if data.pbc is not None else None,
        )
        system = systems_to_torch(atoms)
        neighbor_options = NeighborListOptions(
            cutoff=_CUTOFF, full_list=True, strict=True
        )
        system = get_system_with_neighbor_lists(system, [neighbor_options])
        targets = {
            "energy": get_energy_target_info(
                "energy",
                {"unit": "eV"},
                add_position_gradients=True,
                add_strain_gradients=True,
            )
        }
        mt_output = evaluate_model(
            mt_model, [system], targets=targets, is_training=False
        )

        mt_energy = mt_output["energy"].block().values.detach().squeeze()
        mt_forces = -mt_output["energy"].block().gradient("positions").values.squeeze()
        mt_stress = (
            -mt_output["energy"].block().gradient("strain").values.squeeze()
            / atoms.get_volume()
        )

        nv_output = real_wrapper_cpu.forward(_batch())
        nv_energy = nv_output["energy"].detach().squeeze().item()
        nv_forces = nv_output["forces"].detach()
        nv_stress = nv_output["stress"].detach().squeeze()

        assert abs(nv_energy - mt_energy) < 1e-4, (
            f"Energy mismatch: nvalchemi={nv_energy}, metatrain={mt_energy}"
        )
        assert torch.allclose(nv_forces, mt_forces, atol=1e-4, rtol=1e-4), (
            f"Force mismatch: max |dF|={float((nv_forces - mt_forces).abs().max())}"
        )

        assert torch.allclose(nv_stress, mt_stress, atol=1e-4, rtol=1e-4), (
            f"Stress mismatch: max |dS|={float((nv_stress - mt_stress).abs().max())}"
        )
