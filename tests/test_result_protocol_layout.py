from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


class ResultProtocolLayoutTest(unittest.TestCase):
    def test_no_artifact_is_stored_directly_under_dataset_directory(self) -> None:
        methods = {
            "KD",
            "CRD",
            "ReviewKD",
            "MGD",
            "OFA",
            "LG",
            "ALG",
            "Ours",
            "OursV2",
            "Vanilla",
        }
        for method_dir in RESULTS.iterdir():
            if not method_dir.is_dir() or method_dir.name not in methods:
                continue
            for dataset_dir in method_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                direct_artifacts = [
                    path.name
                    for path in dataset_dir.iterdir()
                    if path.suffix in {".pt", ".json"}
                ]
                self.assertEqual(
                    direct_artifacts,
                    [],
                    f"Artifacts require a protocol-ID directory: {dataset_dir}",
                )

    def test_each_committed_protocol_directory_is_self_contained(self) -> None:
        summaries = sorted(RESULTS.glob("*/*/*/run_summary.json"))
        self.assertEqual(len(summaries), 101)
        for summary_path in summaries:
            run_dir = summary_path.parent
            checkpoint_path = run_dir / "student_best.pt"
            self.assertTrue(checkpoint_path.is_file(), run_dir)
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            method, dataset = summary_path.parts[-4], summary_path.parts[-3]
            self.assertEqual(payload["method"], method)
            self.assertEqual(payload["dataset"], dataset)
            protocol_id = run_dir.name
            args = payload.get("args", {})
            student_epochs = args.get(
                "student_epochs", payload.get("planned_epochs")
            )
            seed = args.get("seed", 1 if method == "Vanilla" else None)
            self.assertIsNotNone(student_epochs, run_dir)
            self.assertIsNotNone(seed, run_dir)
            self.assertIn(str(student_epochs), protocol_id)
            self.assertIn(f"seed{seed}", protocol_id)

    def test_researcher_sync_import_destinations_are_explicit(self) -> None:
        expected = (
            RESULTS / "Ours/cifar100/researcher_sync_v1_300ep_seed1",
            RESULTS / "Ours/flowers102/researcher_sync_v1_300ep_seed1",
            RESULTS / "ALG/flowers102/researcher_sync_v1_300ep_seed1",
        )
        for run_dir in expected:
            self.assertTrue((run_dir / "run_summary.json").is_file(), run_dir)
            self.assertTrue((run_dir / "student_best.pt").is_file(), run_dir)

    def test_table4_table7_import_destinations_are_explicit(self) -> None:
        expected = (
            RESULTS / "Ours/cifar100/table4_kv_independent_researcher_sync_v1_300ep_seed1_k1_v1001",
            RESULTS / "Ours/cifar100/table4_local_patch2_researcher_sync_v1_300ep_seed1_permseed1",
            RESULTS / "Ours/cifar100/table4_token_space_researcher_sync_v1_300ep_seed1",
            RESULTS / "Ours/cifar100/table7_lambda_0p75_researcher_sync_v1_300ep_seed1",
            RESULTS / "Ours/cifar100/table7_lambda_1_researcher_sync_v1_300ep_seed1",
            RESULTS / "OursV2/cifar100/table7_lambda_0_relative_position_v1_300ep_seed1",
            RESULTS / "OursV2/cifar100/table7_lambda_0p5_relative_position_v1_300ep_seed1",
        )
        for run_dir in expected:
            self.assertTrue((run_dir / "run_summary.json").is_file(), run_dir)
            self.assertTrue((run_dir / "student_best.pt").is_file(), run_dir)

    def test_cub200_shared_teacher_import_destinations_are_explicit(self) -> None:
        expected = (
            RESULTS / "LG/cub200/cub200_deit_ti_official_lg_v1_300ep_seed1",
            RESULTS
            / "ALG/cub200/cub200_deit_ti_alg_paper_official_lg_v1_300ep_seed1",
            RESULTS
            / "Ours/cub200/cub200_deit_ti_ours_scratch_teacher_v1_300ep_seed1",
        )
        for run_dir in expected:
            self.assertTrue((run_dir / "run_summary.json").is_file(), run_dir)
            self.assertTrue((run_dir / "student_best.pt").is_file(), run_dir)

    def test_cub200_resnet50_224_import_destinations_are_explicit(self) -> None:
        expected = (
            RESULTS
            / "LG/cub200/cub200_deit_ti_lg_resnet50_224_transfer_adaptation_v1_300ep_seed1",
            RESULTS
            / "ALG/cub200/cub200_deit_ti_alg_resnet50_224_transfer_adaptation_v1_300ep_seed1",
            RESULTS
            / "Ours/cub200/cub200_deit_ti_ours_resnet50_224_transfer_v1_300ep_seed1",
            RESULTS
            / "Vanilla/cub200/cub200_deit_ti_ce_lg_official_b128_300ep_seed1",
            RESULTS
            / "LG/cub200/cub200_deit_ti_lg_resnet50_224_scratch_teacher_ablation_v1_300ep_seed1",
            RESULTS
            / "ALG/cub200/cub200_deit_ti_alg_resnet50_224_scratch_teacher_ablation_v1_300ep_seed1",
            RESULTS
            / "Vanilla/cub200/cub200_deit_ti_ce_ours_current_b64_300ep_seed1",
            RESULTS
            / "Ours/cub200/cub200_deit_ti_ours_resnet50_224_scratch_teacher_ablation_v1_300ep_seed1",
        )
        for run_dir in expected:
            self.assertTrue((run_dir / "run_summary.json").is_file(), run_dir)
            self.assertTrue((run_dir / "student_best.pt").is_file(), run_dir)

        for run_dir in (expected[2], expected[7]):
            self.assertTrue((run_dir / "ours_module_best.pt").is_file(), run_dir)
            self.assertTrue((run_dir / "artifact_manifest.json").is_file(), run_dir)

    def test_cub200_resnet50_teacher_artifact_status_is_explicit(self) -> None:
        transfer_teacher = (
            ROOT / "teachers/checkpoints/cub200_resnet50_224_imagenet1k_v2"
        )
        self.assertTrue(
            (transfer_teacher / "teacher_resnet50_cub200_224_best.pt").is_file()
        )
        self.assertTrue((transfer_teacher / "manifest.json").is_file())
        self.assertTrue((transfer_teacher / "artifact_manifest.json").is_file())
        manifest = json.loads(
            (transfer_teacher / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertIn(
            "80f46b08ea2b2c5398c951268b937f3be0abe47f08bf7617e6dcd4e49a4db82b",
            manifest["teachers"]["cub200"]["equivalent_source_checkpoint_sha256"],
        )
        scratch_teacher = (
            ROOT / "teachers/checkpoints/cub200_resnet50_224_scratch"
        )
        self.assertTrue(
            (
                scratch_teacher
                / "teacher_resnet50_cub200_224_scratch_best.pt"
            ).is_file()
        )
        self.assertTrue((scratch_teacher / "manifest.json").is_file())
        self.assertTrue((scratch_teacher / "artifact_manifest.json").is_file())
        scratch_manifest = json.loads(
            (scratch_teacher / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            scratch_manifest["teachers"]["cub200"][
                "source_checkpoint_sha256"
            ],
            "6307a8289f8ddec5c79e8284af8f07d883d037aeaf936062a6833720e4f74ba7",
        )
        pending = (RESULTS / "PENDING_IMPORTS.md").read_text(encoding="utf-8")
        self.assertNotIn("cub200_resnet50_224_scratch_200ep_seed1", pending)

    def test_build_522_523_import_destinations_are_explicit(self) -> None:
        expected = (
            RESULTS
            / "Ours/cifar100/researcher_sync_v1_batch128_ablation_300ep_seed1",
            RESULTS
            / "Vanilla/cub200/cub200_deit_ti_ce_both_imagenet_pretrained_b128_100ep_seed1",
            RESULTS
            / "LG/cub200/cub200_deit_ti_lg_resnet50_224_both_imagenet_pretrained_b128_100ep_seed1",
            RESULTS
            / "ALG/cub200/cub200_deit_ti_alg_resnet50_224_both_imagenet_pretrained_b128_100ep_seed1",
            RESULTS
            / "Ours/cub200/cub200_deit_ti_ours_resnet50_224_both_imagenet_pretrained_b64_100ep_seed1",
            RESULTS
            / "Ours/cub200/cub200_deit_ti_ours_resnet50_224_both_imagenet_pretrained_b128_100ep_seed1",
        )
        for run_dir in expected:
            self.assertTrue((run_dir / "run_summary.json").is_file(), run_dir)
            self.assertTrue((run_dir / "student_best.pt").is_file(), run_dir)

        for run_dir in expected[-2:]:
            self.assertTrue((run_dir / "ours_module_best.pt").is_file(), run_dir)
            self.assertTrue((run_dir / "artifact_manifest.json").is_file(), run_dir)

    def test_all_36_table1_baseline_destinations_are_explicit(
        self,
    ) -> None:
        teacher = ROOT / "teachers/checkpoints/cub200_table1_resnet56_32"
        manifest = json.loads(
            (teacher / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["teachers"]["cub200"]["sha256"],
            "06f75192b1c108c89e480843cb4f72dfb28aa762d7b11e7ac327333dd54b51f5",
        )
        self.assertTrue(
            (teacher / "teacher_resnet56_cub200_32_best.pt").is_file()
        )
        method_settings = (
            ("Vanilla", "vanilla", 128),
            ("LG", "lg", 128),
            ("ALG", "alg", 128),
            ("Ours", "ours", 64),
            ("Ours", "ours", 128),
        )
        student_builds = {
            "deit_ti": {"vanilla": 548, "lg": 543, "alg": 543, "ours": 547},
            "convit_ti": {"all": 551},
            "cvt_13": {"all": 552},
            "pit_ti": {"all": 553},
            "pvtv2_b0": {"all": 554},
            "t2t_vit_7": {"all": 555},
            "t2t_vit_14": {"all": 556},
        }
        for student, build_map in student_builds.items():
            for method, method_key, batch in method_settings:
                expected_build = build_map.get(method_key, build_map.get("all"))
                self.assertIsNotNone(expected_build)
                protocol = (
                    f"table1_cub200_{student}_{method_key}_b{batch}"
                    "_full_300ep_seed1"
                )
                run_dir = RESULTS / method / "cub200" / protocol
                self.assertTrue((run_dir / "run_summary.json").is_file())
                self.assertTrue((run_dir / "student_best.pt").is_file())
                summary = json.loads(
                    (run_dir / "run_summary.json").read_text(encoding="utf-8")
                )
                self.assertEqual(summary["actual_epochs"], 300)
                self.assertEqual(summary["h200_build"], expected_build)
                if method == "Vanilla":
                    self.assertIsNone(summary["teacher"])
                else:
                    self.assertEqual(
                        summary["teacher"]["sha256"],
                        manifest["teachers"]["cub200"]["sha256"],
                    )

    def test_consolidated_table_uses_only_current_reporting_results(self) -> None:
        root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
        results_readme = (RESULTS / "README.md").read_text(encoding="utf-8")
        expected_rows = (
            "| KD | Logits | 69.10 | 48.95 | 62.79 |  |",
            "| CRD | Pooled contrastive | 68.59 | 49.06 | 79.85 |  |",
            "| ReviewKD | Projected fusion | 75.65 | 61.88 | 82.75 |  |",
            "| MGD | Masked reconstruction | 75.68 | 54.66 | 81.81 |  |",
            "| OFA | Logit-space projection | 67.73 | 46.41 | 78.03 |  |",
            "| LG | Direct match (static) |  |  |  | 46.93 |",
            "| ALG | Scheduled match (static) |  | 73.15 | 83.54 | 49.02 |",
            "| **Ours** | **Grid-space, learnable** | **82.90** | **74.81** | **81.95\\*** | **48.72** |",
        )
        for row in expected_rows:
            self.assertIn(row, root_readme)
            self.assertIn(row, results_readme)

        pending = (RESULTS / "PENDING_IMPORTS.md").read_text(encoding="utf-8")
        self.assertIn("Ours | Chaoyang", pending)
        self.assertIn("researcher_sync_v1_300ep_seed1", pending)
        self.assertIn("81.95%", pending)


if __name__ == "__main__":
    unittest.main()
