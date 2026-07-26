# CUB-200 random-init ResNet50-224 teacher

Verified H200 build-519 teacher for the isolated ResNet50-224 scratch
ablation. The committed checkpoint retains every source model tensor and all
metadata, but omits `optimizer_state` to remain below GitHub's 100 MB
single-file limit. Exact source and committed hashes are recorded in
`artifact_manifest.json`.

This teacher is not used by either ResNet56-based CUB family.
