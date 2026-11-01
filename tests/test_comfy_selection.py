"""Offline integrity/admission checks for the pending M2.1 candidate; no model bytes."""

import copy
import hashlib
import json
import unittest
from pathlib import Path

from cineforge.devices import Backend, GPUDevice, ModelSpec, Precision, select_device
from cineforge.selection import SelectionError, require_selection_approval, selection_subject

ROOT = Path(__file__).resolve().parents[1] / "config" / "comfy"
LEDGER = ROOT / "wan21-t2v-1.3b.json"
ASSETS = {
    "split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors": (
        2838303560, "be531024cd9018cb5b48c40cfbb6a6191645b1c792eb8bf4f8c1c6e10f924dc5",  # pragma: allowlist secret — public provenance hash
    ),
    "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors": (
        6735906897, "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68",  # pragma: allowlist secret — public provenance hash
    ),
    "split_files/vae/wan_2.1_vae.safetensors": (
        253815318, "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b",  # pragma: allowlist secret — public provenance hash
    ),
}


def _unique_keys(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate selection JSON key")
    return result


class ComfySelectionTests(unittest.TestCase):
    def setUp(self):
        self.selection = json.loads(LEDGER.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
        self.workflow = self.selection["workflow"]
        self.assertEqual(self.workflow["path"], "wan21-t2v-1.3b-api.json")
        self.graph = json.loads((ROOT / self.workflow["path"]).read_text(encoding="utf-8"))

    def test_immutable_revisions_and_exact_asset_metadata(self):
        selection = self.selection
        self.assertEqual(selection["schema_version"], 1)
        self.assertEqual(selection["task_id"], "M2.1")
        pins = {
            selection["model"]["revision"]: "123acf1cc74bccbb9bfff8ac1ee72edc08c2341d",  # pragma: allowlist secret — public provenance hash
            selection["model"]["upstream_revision"]: "37ec512624d61f7aa208f7ea8140a131f93afc9a",  # pragma: allowlist secret — public provenance hash
            selection["runner"]["revision"]: "6d352802d8f8b9133f664a1757d6c6d26390c335",  # pragma: allowlist secret — public provenance hash
            self.workflow["source_revision"]: "f9431bb000ce792094ff345446e22cac1ea6cef3",  # pragma: allowlist secret — public provenance hash
        }
        for actual, expected in pins.items():
            self.assertEqual(actual, expected)
            self.assertRegex(actual, r"^[0-9a-f]{40}$")
        assets = selection["model"]["files"]
        self.assertEqual(len(assets), len(ASSETS))
        self.assertEqual({asset["path"] for asset in assets}, set(ASSETS))
        for asset in assets:
            self.assertEqual((asset["size_bytes"], asset["sha256"]), ASSETS[asset["path"]])
            self.assertEqual(asset["revision"], selection["model"]["revision"])
            self.assertEqual(asset["url"], "https://huggingface.co/Comfy-Org/Wan-2.1/resolve/"
                             + asset["revision"] + "/" + asset["path"])
            self.assertIn("not downloaded", asset["hash_verification"])
        self.assertEqual(selection["model"]["total_weight_bytes"], 9828025775)
        self.assertEqual(sum(a["size_bytes"] for a in assets), 9828025775)

    def test_exact_graph_digest_and_retained_workflow_license(self):
        canonical = json.dumps(self.graph, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        self.assertEqual(digest, self.workflow["canonical_sha256"])
        self.assertEqual(digest, "5f6860c7c9817023b37c0bbe707f7f50d922964a3b729cdac2d2831bf822c5a8")  # pragma: allowlist secret — public provenance hash
        self.assertEqual(self.workflow["source_raw_sha256"],
                         "436b9c074b677011b0f50a9feb8c4b4128abbbd9e4acaf34d8eeb80efbd74a64")  # pragma: allowlist secret — public provenance hash
        self.assertEqual(self.workflow["license_path"], "WAN21-WORKFLOW-LICENSE")
        license_bytes = (ROOT / self.workflow["license_path"]).read_bytes()
        self.assertEqual(hashlib.sha256(license_bytes).hexdigest(),
                         self.workflow["license_sha256"])
        self.assertIn(b"Permission to use, copy, modify", license_bytes)
        self.assertIn(b"any purpose with or without fee", license_bytes)

    def test_only_pinned_core_classes_and_resolvable_graph_links(self):
        self.assertEqual(set(self.graph), {"3", "6", "7", "8", "37", "38", "39", "40", "47", "48"})
        expected = {"KSampler", "CLIPTextEncode", "VAEDecode", "UNETLoader", "CLIPLoader",
                    "VAELoader", "EmptyHunyuanLatentVideo", "SaveWEBM", "ModelSamplingSD3"}
        self.assertEqual(set(self.workflow["allowed_classes"]), expected)
        self.assertEqual({node["class_type"] for node in self.graph.values()}, expected)
        self.assertEqual(self.selection["runner"]["custom_nodes"], [])
        for node in self.graph.values():
            self.assertEqual(set(node), {"class_type", "inputs"})
            for value in node["inputs"].values():
                if isinstance(value, list):
                    self.assertEqual(len(value), 2)
                    self.assertIn(value[0], self.graph)
                    self.assertIs(type(value[1]), int)
                    self.assertEqual(value[1], 0)
        self.assertEqual(self.graph["47"]["inputs"]["images"], ["8", 0])
        self.assertEqual(self.graph["8"]["inputs"], {"samples": ["3", 0], "vae": ["39", 0]})

    def test_bindings_output_and_loader_files_match_ledger(self):
        self.assertEqual(self.workflow["bindings"], {
            "prompt": {"node_id": "6", "input_name": "text"},
            "seed": {"node_id": "3", "input_name": "seed"},
        })
        for kind, binding in self.workflow["bindings"].items():
            value = self.graph[binding["node_id"]]["inputs"][binding["input_name"]]
            self.assertIs(type(value), str if kind == "prompt" else int)
        self.assertEqual(self.workflow["output_node"], "47")
        self.assertEqual(self.workflow["output_kind"], "images")
        self.assertEqual(self.workflow["output_index"], 0)
        self.assertEqual(self.workflow["media_type"], "video/webm")
        self.assertEqual(self.graph["47"]["class_type"], "SaveWEBM")
        self.assertEqual(self.graph["47"]["inputs"]["codec"], "vp9")
        self.assertEqual(self.graph["40"]["inputs"],
                         {"batch_size": 1, "height": 480, "length": 33, "width": 832})
        self.assertEqual(self.workflow["duration_seconds"],
                         self.graph["40"]["inputs"]["length"] / self.graph["47"]["inputs"]["fps"])
        loaders = {"37": "unet_name", "38": "clip_name", "39": "vae_name"}
        loaded = {self.graph[node]["inputs"][field] for node, field in loaders.items()}
        self.assertEqual(loaded, {Path(path).name for path in ASSETS})

    def test_declared_cuda_target_does_not_claim_measured_or_metal_support(self):
        declared = self.selection["model"]
        model = ModelSpec(
            model_id=declared["model_id"], revision=declared["revision"], task=declared["task"],
            allowed_backends=frozenset(Backend(x) for x in declared["allowed_backends"]),
            minimum_memory_mib=declared["minimum_memory_mib"], license=declared["license"],
            required_precision=Precision(declared["required_precision"]),
            commercial_use=declared["commercial_use"],
        )
        self.assertEqual(model.allowed_backends, frozenset({Backend.CUDA}))
        self.assertEqual(model.minimum_memory_mib, 24576)
        candidate = GPUDevice("declared-4090", Backend.CUDA, 24576,
                              frozenset({Precision.FP16}))
        self.assertEqual(select_device(model, [candidate]), candidate)
        self.assertIs(self.selection["hardware"]["measured"], False)
        self.assertIn("not a measured minimum", declared["memory_note"])
        self.assertIn("Metal compatibility is not claimed", " ".join(
            self.selection["hardware"]["limits"]))

    def test_owner_approval_binds_selection_without_claiming_live_readiness(self):
        selection = self.selection
        self.assertEqual(selection["state"], "selection_approved")
        approval = selection["approval"]
        self.assertEqual(approval["execution_target"]["status"], "approved")
        record = approval["model_workflow"]
        self.assertEqual(record["status"], "approved")
        self.assertEqual(record["evidence"],
                         "https://github.com/richardo920/cineforge/issues/40#issuecomment-6553169092")
        self.assertTrue(record["approved_by"])
        for flag in ("model_approved", "workflow_approved", "hardware_approved"):
            self.assertIs(record[flag], True)
        self.assertIs(approval["weight_downloads"], True)
        self.assertEqual(approval["spend_limit_usd"], 3)
        self.assertIs(approval["live_execution"], False)
        self.assertIs(approval["visual_review"], False)
        self.assertIs(selection["readiness"]["live_ready"], False)
        self.assertTrue(selection["readiness"]["blockers"])
        self.assertTrue(all("M2.4" not in blocker
                            for blocker in selection["readiness"]["blockers"]))
        self.assertEqual(self.workflow["artifact_license"], "LicenseRef-Generated-Unreviewed")
        self.assertEqual(len(selection["rights_ledger"]), 5)
        require_selection_approval(selection, self.graph)
        subject = json.dumps(selection_subject(selection), sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.assertEqual(hashlib.sha256(subject).hexdigest(), record["approved_subject_sha256"])

    def test_missing_or_incomplete_approval_is_rejected(self):
        cases = [
            ("status", "pending"), ("evidence", None), ("evidence", "https://example.com/"),
            ("approved_by", " "), ("model_approved", False), ("workflow_approved", False),
            ("hardware_approved", False), ("approved_subject_sha256", "0" * 64),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                changed = copy.deepcopy(self.selection)
                changed["approval"]["model_workflow"][key] = value
                with self.assertRaises(SelectionError):
                    require_selection_approval(changed, self.graph)
        for value in (None, False, -1, 0, float("nan"), float("inf")):
            with self.subTest(budget=value):
                changed = copy.deepcopy(self.selection)
                changed["approval"]["spend_limit_usd"] = value
                with self.assertRaises(SelectionError):
                    require_selection_approval(changed, self.graph)
        for key in ("model_workflow", "weight_downloads", "spend_limit_usd"):
            changed = copy.deepcopy(self.selection)
            del changed["approval"][key]
            with self.assertRaises(SelectionError):
                require_selection_approval(changed, self.graph)
        changed = copy.deepcopy(self.selection)
        changed["approval"]["weight_downloads"] = False
        with self.assertRaises(SelectionError):
            require_selection_approval(changed, self.graph)

    def test_changed_model_assets_hardware_workflow_or_budget_reject_old_approval(self):
        for section, key, value in (
            ("model", "revision", "0" * 40), ("model", "model_id", "different"),
            ("model", "allowed_backends", ["metal"]), ("model", "license", "different"),
            ("hardware", "gpu", "different"), ("hardware", "vram_gib", 12),
            ("workflow", "output_node", "28"), ("approval", "spend_limit_usd", 4),
        ):
            with self.subTest(section=section, key=key):
                changed = copy.deepcopy(self.selection)
                changed[section][key] = value
                with self.assertRaises(SelectionError):
                    require_selection_approval(changed, self.graph)
        for key, value in (("size_bytes", 1), ("sha256", "0" * 64),
                           ("url", "https://example.com/changed")):
            changed = copy.deepcopy(self.selection)
            changed["model"]["files"][0][key] = value
            with self.assertRaises(SelectionError):
                require_selection_approval(changed, self.graph)
        graph = copy.deepcopy(self.graph)
        graph["3"]["inputs"]["steps"] = 1
        with self.assertRaises(SelectionError):
            require_selection_approval(self.selection, graph)
        changed = copy.deepcopy(self.selection)
        canonical = json.dumps(graph, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        changed["workflow"]["canonical_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        with self.assertRaises(SelectionError):
            require_selection_approval(changed, graph)


if __name__ == "__main__":
    unittest.main()
