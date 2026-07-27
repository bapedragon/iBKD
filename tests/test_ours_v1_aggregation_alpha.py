import json

import torch

from methods.Ours.aggregation_alpha.extract_alpha import RunSpec, extract_run


def test_extract_run_uses_softmax_and_one_based_top3(tmp_path):
    run_dir = tmp_path / "results/Ours/cifar100/example"
    run_dir.mkdir(parents=True)
    raw_weights = torch.arange(36, dtype=torch.float32).reshape(3, 12)
    alpha = torch.softmax(raw_weights, dim=-1)
    torch.save(
        {
            "method": "Ours",
            "student": "deit_ti",
            "dataset": "cifar100",
            "epoch": 17,
            "accuracy": 81.25,
            "ours": {"aggregation.weights": raw_weights},
        },
        run_dir / "student_best.pt",
    )
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "dataset": "cifar100",
                "best_top1": 81.25,
                "aggregation_weights": alpha.tolist(),
            }
        ),
        encoding="utf-8",
    )
    spec = RunSpec(
        dataset="cifar100",
        display_name="CIFAR-100",
        relative_dir="results/Ours/cifar100/example",
    )

    result = extract_run(tmp_path, spec)

    assert result["shape"] == [3, 12]
    assert result["checkpoint_epoch"] == 17
    assert result["checkpoint_accuracy"] == 81.25
    assert result["summary_crosscheck_max_abs_diff"] <= 1e-6
    assert result["stages"][0]["top3"][0]["student_block"] == 12
    for stage in result["stages"]:
        assert abs(stage["sum"] - 1.0) <= 1e-6
