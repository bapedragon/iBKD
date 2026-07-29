# CUB-200 DeiT-Ti LG val-selected / fixed 36.40 Teacher Issue

## 제목

```text
[Request]: 박철현 CUB-200 DeiT-Ti LG val-best 선택 + fixed 36.40 Teacher + test 1회
```

## 개인 / 연구실 계정 사용자 ID

```text
bapedragon / kau-aimslab
```

## 제출 계정

```text
bapedragon 또는 kau-aimslab 중 실제 제출하는 계정
```

## 실행할 코드의 GitHub 링크

```text
https://github.com/bapedragon/IBAM_KD_H200_V2.git
```

## 코드 실행 명령어

```text
python methods/table1_cub200/train.py --full-run --student deit_ti --method lg --batch-size 128 --selection-protocol val_then_test_once --teacher-contract fixed_build543 --val-per-class 6 --val-split-seed 2027 --teacher-root teachers/checkpoints/cub200_table1_resnet56_32 --data-dir /app/output/cub200_deit_ti_lg_b128_valselect_fixed36p4_seed1/data/cub200 --output-dir /app/output/cub200_deit_ti_lg_b128_valselect_fixed36p4_seed1 --run-name cub200_deit_ti_lg_b128_valselect_fixed_teacher36p4_300ep_seed1_split2027 --num-workers 4
```

## 이미지 / 언어 / GPU

```text
pytorch/pytorch:latest
Python
MIG 7개
```

## 요청 내용

```text
CUB-200-2011 official train 5,994장에서 클래스별 6장씩 총 1,200장을 deterministic stratified validation으로 분리한다. 실제 학습 train은 4,794장, validation은 1,200장, untouched official test는 5,794장이다. split seed는 2027이며 validation image ID 목록과 SHA-256을 저장한다.

scratch DeiT-Ti를 224×224, batch 128, 300 epochs, seed 1, FP32로 LG 학습한다. Teacher는 새로 학습하지 않고 기존 CUB Table-1 Build 543 scratch ResNet56-32 checkpoint를 고정 재사용한다. Teacher Top-1은 36.40%, epoch 275, SHA-256은 06f75192b1c108c89e480843cb4f72dfb28aa762d7b11e7ac327333dd54b51f5다.

매 epoch에는 validation Top-1만 평가해 val-best checkpoint를 선택한다. official test는 학습 loop에서 평가하지 않고, 300 epochs 종료 후 val-best checkpoint를 다시 불러와 정확히 한 번만 평가한다. 최종 보고값은 final_test_top1이며 best_validation_top1은 선택 근거로 별도 기록한다.

주의: Student checkpoint 선택은 official test와 분리되지만, 재사용하는 36.40 Teacher 자체는 과거 official test 최고값으로 선택된 legacy checkpoint다. 따라서 이 실행은 student-level test-selection bias만 제거하며 전체 pipeline이 완전히 test-independent한 것은 아니다.
```

## 정상 시작 확인 문구

```text
[VAL_SPLIT] seed=2027 val_per_class=6 train=4794 validation=1200 test=5794 sha256=...
[TEST_ISOLATION] official_test_in_training_loop=False final_test_evaluations=1
[TEACHER_SELECTION_CAVEAT] teacher=fixed_build543_36.40 selected_on=legacy_best_test student_selected_on=validation
[FIXED_TEACHER_TABLE1_CUB200] h200_build=543 root=... sha256=06f75192b1c108c89e480843cb4f72dfb28aa762d7b11e7ac327333dd54b51f5
```

## 정상 완료 확인 문구

```text
[FINAL_TEST_ONCE][LG] selected_epoch=... best_val_top1=...% test_top1=...% evaluations=1
[FINAL_RESULT] method=LG best_val_top1=...% final_test_top1=...% test_evaluations=1
[DONE] Table-1 CUB student task completed successfully.
```

## 완료 판정

```text
300 epochs가 완료되고 summary.json의 selection_protocol=val_then_test_once, selection_metric=validation_top1, test_evaluations=1이어야 한다. validation_split은 train=4794, validation=1200, split_seed=2027을 기록해야 한다. Teacher SHA-256은 고정 Build 543 hash와 정확히 같아야 한다.
```

## 첨부 파일

```text
/app/output/cub200_deit_ti_lg_b128_valselect_fixed36p4_seed1/cub200_deit_ti_lg_b128_valselect_fixed_teacher36p4_300ep_seed1_split2027/summary.json
/app/output/cub200_deit_ti_lg_b128_valselect_fixed36p4_seed1/cub200_deit_ti_lg_b128_valselect_fixed_teacher36p4_300ep_seed1_split2027/validation_split.json
/app/output/cub200_deit_ti_lg_b128_valselect_fixed36p4_seed1/cub200_deit_ti_lg_b128_valselect_fixed_teacher36p4_300ep_seed1_split2027/student_best.pt
```
