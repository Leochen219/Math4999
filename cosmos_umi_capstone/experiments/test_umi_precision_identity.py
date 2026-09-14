"""Actual Torch state omitted by state_dict must still bind experiment identity."""
import copy
import dataclasses
import tempfile
import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:
    torch = None

import umi_precision_runtime as api
from umi_precision_identity import fingerprint
from umi_precision_official import TorchOps
import test_umi_precision_runtime as fixtures


class IdentityGraphTests(unittest.TestCase):
    def test_fingerprint_handles_owner_back_reference_and_tracks_state(self):
        owner = SimpleNamespace(mode="initial")
        child = SimpleNamespace(owner=owner, value=1)
        owner.child = child
        before = fingerprint(owner)
        owner.mode = "changed"
        self.assertNotEqual(before, fingerprint(owner))

    def test_fingerprint_handles_slots_dataclass_and_tracks_fields(self):
        @dataclasses.dataclass(slots=True)
        class SlotRecord:
            value: int
            label: str

        record = SlotRecord(1, "clean")
        before = fingerprint(record)
        record.value = 2
        self.assertNotEqual(before, fingerprint(record))

    def test_fingerprint_is_stable_across_deepcopy_and_distinguishes_shared_objects(self):
        class Node:
            def __init__(self, value):
                self.value = value
                self.link = None

        shared = Node("same")
        shared_root = {"left": shared, "right": shared}
        independent_root = {"left": Node("same"), "right": Node("same")}
        self.assertEqual(fingerprint(shared_root), fingerprint(copy.deepcopy(shared_root)))
        self.assertNotEqual(fingerprint(shared_root), fingerprint(independent_root))

    def test_fingerprint_distinguishes_different_ancestor_backreferences(self):
        class Node:
            def __init__(self):
                self.child = None
                self.back = None

        def graph(cross):
            first, second = Node(), Node()
            first.child, second.child = Node(), Node()
            if cross:
                first.child.back, second.child.back = second, first
            else:
                first.child.back, second.child.back = first, second
            return {"first": first, "second": second}

        self.assertNotEqual(fingerprint(graph(True)), fingerprint(graph(False)))

    def test_set_fingerprint_is_address_independent_for_stateful_objects(self):
        class Stateful:
            def __init__(self, value):
                self.value = value
                self.shared = None
            def __repr__(self):
                return f"Stateful(address={hex(id(self))}, value={self.value})"
            def __hash__(self):
                return id(self)

        first, second = Stateful(1), Stateful(2)
        first.shared = second
        second.shared = first
        original = {"items": {first, second}}
        self.assertEqual(fingerprint(original), fingerprint(copy.deepcopy(original)))


@unittest.skipIf(torch is None, "requires installed Torch CPU runtime")
class CompleteModuleIdentityTests(unittest.TestCase):
    def module(self):
        class Compute(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.000123]))
                self.register_buffer("_timestep_frequencies", torch.tensor([1., 2.], dtype=torch.float64), persistent=False)
                self.extra = {"compute_policy": "fixed"}

            def get_extra_state(self):
                return self.extra

            def set_extra_state(self, state):
                self.extra = state

        return torch.nn.Sequential(Compute())

    def test_nonpersistent_full_bytes_dtype_shape_and_names_bind_identity(self):
        for mutation in (
            lambda child: child._timestep_frequencies.add_(1e-10),
            lambda child: setattr(child, "_timestep_frequencies", child._timestep_frequencies.float()),
            lambda child: setattr(child, "_timestep_frequencies", child._timestep_frequencies.reshape(1, 2)),
            lambda child: child.register_buffer("extra_frequency", child._timestep_frequencies, persistent=False),
        ):
            with self.subTest(mutation=mutation):
                module = self.module()
                before = fingerprint(module)
                mutation(module[0])
                self.assertNotEqual(before, fingerprint(module))

    def test_training_and_extra_compute_state_bind_identity(self):
        module = self.module()
        before = fingerprint(module)
        module[0].eval()
        self.assertNotEqual(before, fingerprint(module))
        before = fingerprint(module)
        module[0].extra["compute_policy"] = "changed"
        self.assertNotEqual(before, fingerprint(module))

    def test_resume_rejects_nonpersistent_model_and_decoder_buffer_changes(self):
        alphas = [.0001, .0003, .001, .003, .01, .03]
        for owner in ("model", "decoder"):
            with self.subTest(owner=owner), tempfile.TemporaryDirectory() as root:
                runtime = fixtures.FakeRuntime(api)
                runtime.model_state = {"model": self.module(), "decoder": self.module()}
                inputs = fixtures.RuntimeTests.inputs(SimpleNamespace(api=api))
                result = api.run_precision_experiment(runtime, inputs, root, alphas=alphas)
                self.assertEqual(result["status"], "complete")
                count = len(runtime.calls)
                runtime.model_state[owner][0]._timestep_frequencies.add_(1e-10)
                with self.assertRaisesRegex(ValueError, "strict resume"):
                    api.run_precision_experiment(runtime, inputs, root, alphas=alphas, resume=True)
                self.assertEqual(len(runtime.calls), count)

    def test_clone_checks_nonpersistent_storage_values_and_registry(self):
        original = torch.nn.Sequential(torch.nn.Module())
        original[0].register_buffer("_timestep_frequencies", torch.tensor([1., 2.], dtype=torch.float64), persistent=False)
        ops = TorchOps()
        ops.ensure_independent(original, copy.deepcopy(original))
        for mutation, message in (
            (lambda clone: setattr(clone[0], "_timestep_frequencies", original[0]._timestep_frequencies), "aliases"),
            (lambda clone: clone[0]._timestep_frequencies.add_(1e-10), "values"),
            (lambda clone: delattr(clone[0], "_timestep_frequencies"), "keys"),
        ):
            with self.subTest(message=message):
                clone = copy.deepcopy(original)
                mutation(clone)
                with self.assertRaisesRegex(api.EvidenceError, message):
                    ops.ensure_independent(original, clone)


if __name__ == "__main__":
    unittest.main()
