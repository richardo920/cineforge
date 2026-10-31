"""Synthetic capability contracts; these fixtures make no claims about real hardware."""

import dataclasses
import itertools
import math
import unittest

from cineforge.devices import (
    Backend,
    DeviceSelectionError,
    GPUDevice,
    ModelSpec,
    Precision,
    compatibility,
    select_device,
)


def model(**changes):
    fields = {
        "model_id": "synthetic/model",
        "revision": "synthetic-v1",
        "task": "text-to-video",
        "allowed_backends": frozenset({Backend.CUDA}),
        "minimum_memory_mib": 8192,
        "license": "LicenseRef-SyntheticFixture",
    }
    fields.update(changes)
    return ModelSpec(**fields)


class CapabilityContractTests(unittest.TestCase):
    def test_all_backends_are_explicitly_supported_by_synthetic_fixtures(self):
        for backend in Backend:
            with self.subTest(backend=backend):
                device = GPUDevice(f"synthetic-{backend.value}", backend, 8192)
                spec = model(allowed_backends=frozenset({backend}))
                self.assertEqual(compatibility(device, spec), ())
                self.assertIs(select_device(spec, [device]), device)

    def test_exact_memory_boundary_and_zero_capacity_requirement(self):
        device = GPUDevice("cpu", Backend.CPU, 0)
        spec = model(allowed_backends=frozenset({Backend.CPU}), minimum_memory_mib=0)
        self.assertIs(select_device(spec, [device]), device)
        self.assertEqual(compatibility(GPUDevice("gpu", Backend.CUDA, 8192), model()), ())

    def test_memory_requires_exact_nonnegative_integers(self):
        for value in [-1, True, False, 8192.0, math.nan, math.inf, -math.inf, "8192", None]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "nonnegative integer"):
                    GPUDevice("gpu", Backend.CUDA, value)
                with self.assertRaisesRegex(ValueError, "nonnegative integer"):
                    model(minimum_memory_mib=value)

    def test_text_metadata_rejects_invalid_values(self):
        for value in ["", " ", " gpu", "gpu ", "gpu\n", "gpu\0", "gpu\x7f", "gpu\x85name", "gpu\ud800name", 1, None]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    GPUDevice(value, Backend.CUDA, 8192)
                for field in ["model_id", "revision", "task", "license"]:
                    with self.assertRaises(ValueError):
                        model(**{field: value})

    def test_backend_and_precision_are_typed_declarations(self):
        for backend in ["cuda", "nvidia", Precision.FP32, None]:
            with self.assertRaisesRegex(ValueError, "Backend"):
                GPUDevice("gpu", backend, 8192)
        for values in [frozenset(), frozenset({"cuda"}), {Backend.CUDA}, [Backend.CUDA], None]:
            with self.assertRaisesRegex(ValueError, "nonempty frozenset"):
                model(allowed_backends=values)
        for values in [frozenset(), frozenset({"fp32"}), {Precision.FP32}, None]:
            with self.assertRaisesRegex(ValueError, "nonempty frozenset"):
                GPUDevice("gpu", Backend.CUDA, 8192, values)
        with self.assertRaisesRegex(ValueError, "Precision"):
            model(required_precision="fp16")

    def test_common_floating_revisions_rejected(self):
        for revision in ["main", "master", "latest", "HEAD", "default", "stable", "nightly"]:
            with self.subTest(revision=revision):
                with self.assertRaisesRegex(ValueError, "pinned revision"):
                    model(revision=revision)

    def test_rights_are_declared_metadata_not_hardware_compatibility_gates(self):
        device = GPUDevice("gpu", Backend.CUDA, 8192)
        for declaration in [None, False, True]:
            spec = model(commercial_use=declaration)
            self.assertIs(spec.commercial_use, declaration)
            self.assertIs(select_device(spec, [device]), device)
        for value in [0, 1, "approved", [], math.nan]:
            with self.assertRaisesRegex(ValueError, "bool or None"):
                model(commercial_use=value)

    def test_capabilities_are_immutable(self):
        device = GPUDevice("gpu", Backend.CUDA, 8192)
        spec = model()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            device.memory_mib = 16384
        with self.assertRaises(dataclasses.FrozenInstanceError):
            spec.revision = "changed"
        with self.assertRaises(AttributeError):
            device.supported_precisions.add(Precision.FP16)


class SelectionTests(unittest.TestCase):
    def test_more_memory_never_overrides_incompatible_backend(self):
        device = GPUDevice("large-rocm", Backend.ROCM, 1024 * 1024)
        with self.assertRaisesRegex(DeviceSelectionError, "backend rocm is not allowed"):
            select_device(model(), [device])

    def test_insufficient_memory_is_explicit(self):
        with self.assertRaisesRegex(DeviceSelectionError, "insufficient memory: 8191 MiB < 8192"):
            select_device(model(), [GPUDevice("small", Backend.CUDA, 8191)])

    def test_missing_devices_is_explicit(self):
        with self.assertRaisesRegex(DeviceSelectionError, "no devices available"):
            select_device(model(), [])

    def test_precision_is_required_even_with_sufficient_capacity(self):
        spec = model(required_precision=Precision.BF16)
        fp32 = GPUDevice("fp32", Backend.CUDA, 16384)
        bf16 = GPUDevice("bf16", Backend.CUDA, 8192, frozenset({Precision.FP32, Precision.BF16}))
        with self.assertRaisesRegex(DeviceSelectionError, "precision bf16 is not supported"):
            select_device(spec, [fp32])
        self.assertIs(select_device(spec, [fp32, bf16]), bf16)

    def test_reports_all_incompatibilities_for_each_device(self):
        reasons = compatibility(
            GPUDevice("wrong", Backend.CPU, 1), model(required_precision=Precision.FP16),
        )
        self.assertEqual(len(reasons), 3)
        self.assertIn("backend cpu", reasons[0])
        self.assertIn("insufficient memory", reasons[1])
        self.assertIn("precision fp16", reasons[2])
        devices = [GPUDevice("z", Backend.METAL, 1), GPUDevice("a", Backend.CUDA, 2)]
        messages = []
        for order in itertools.permutations(devices):
            with self.assertRaises(DeviceSelectionError) as context:
                select_device(model(), order)
            messages.append(str(context.exception))
        self.assertEqual(messages[0], messages[1])
        self.assertIn("a: insufficient memory", messages[0])
        self.assertIn("z: backend metal", messages[0])

    def test_selection_is_independent_of_input_order(self):
        spec = model(allowed_backends=frozenset(Backend))
        devices = [
            GPUDevice("z", Backend.CPU, 8192),
            GPUDevice("b", Backend.CUDA, 8192),
            GPUDevice("a", Backend.CUDA, 8192),
            GPUDevice("small", Backend.CUDA, 8191),
            GPUDevice("big", Backend.CUDA, 16384),
        ]
        for order in itertools.permutations(devices):
            self.assertEqual(select_device(spec, order).device_id, "z")
        # Equal memory/backend ties use stable identity, independent of input order.
        for order in itertools.permutations(devices[1:]):
            self.assertEqual(select_device(spec, order).device_id, "a")

    def test_inventory_cannot_hide_duplicate_ids_after_a_compatible_device(self):
        for devices in [
            [GPUDevice("same", Backend.CUDA, 8192)] * 2,
            [GPUDevice("same", Backend.CUDA, 8192), GPUDevice("same", Backend.CPU, 16384)],
        ]:
            with self.assertRaisesRegex(DeviceSelectionError, "duplicate device IDs"):
                select_device(model(), iter(devices))

    def test_invalid_inventory_or_model_refused(self):
        with self.assertRaisesRegex(DeviceSelectionError, "GPUDevice"):
            select_device(model(), ["gpu"])
        with self.assertRaisesRegex(DeviceSelectionError, "ModelSpec"):
            select_device(None, [])
        with self.assertRaisesRegex(ValueError, "GPUDevice and ModelSpec"):
            compatibility("gpu", model())


if __name__ == "__main__":
    unittest.main()
