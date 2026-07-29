# CUB-200 CvT-13 Ours val-selected / fixed 36.40 Teacher Issue

## 제목

```text
[Request]: 박철현 CUB-200 CvT-13 Ours val-best 선택 + fixed 36.40 Teacher + test 1회
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
python methods/table1_cub200/train.py --full-run --student cvt_13 --method ours --batch-size 128 --selection-protocol val_then_test_once --teacher-contract fixed_build543 --val-per-class 6 --val-split-seed 2027 --teacher-root teachers/checkpoints/cub200_table1_resnet56_32 --data-dir /app/output/cub200_cvt_13_ours_b128_valselect_fixed36p4_seed1/data/cub200 --output-dir /app/output/cub200_cvt_13_ours_b128_valselect_fixed36p4_seed1 --run-name cub200_cvt_13_ours_b128_valselect_fixed_teacher36p4_300ep_seed1_split2027 --num-workers 4
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

공식 LG tiny-transformers 코드와 고정 설정의 scratch CvT-13을 224×224, batch 128, 300 epochs, seed 1, FP32로 Ours v1 학습한다. CvT-13의 전체 13개 block을 사용하는 Ours all-block aggregation, channel/deformable attention, convolutional cross-attention, lambda 0.5 및 기존 ALG 기반 guidance controller를 유지한다. Teacher는 기존 CUB Table-1 Build 543 scratch ResNet56-32 checkpoint(36.40%, epoch 275)를 고정 재사용한다. SHA-256은 06f75192b1c108c89e480843cb4f72dfb28aa762d7b11e7ac327333dd54b51f5다.

매 epoch에는 validation Top-1만 평가하고 official test는 사용하지 않는다. 300 epochs 종료 후 val-best checkpoint를 불러와 official test를 정확히 한 번 평가한다. final_test_top1을 최종 결과로 보고하고 best_validation_top1을 별도 기록한다.

주의: Student의 epoch 선택에는 test를 사용하지 않지만 36.40 Teacher는 legacy best-test checkpoint다. 따라서 제거되는 것은 Student의 test-selection bias이며 전체 pipeline이 완전히 test-independent한 것은 아니다.
```

## 정상 시작 확인 문구

```text
[TASK] student=CvT-13 method=OURS batch=128 mode=full_300ep
[VAL_SPLIT] seed=2027 val_per_class=6 train=4794 validation=1200 test=5794 sha256=...
[TEST_ISOLATION] official_test_in_training_loop=False final_test_evaluations=1
[TEACHER_SELECTION_CAVEAT] teacher=fixed_build543_36.40 selected_on=legacy_best_test student_selected_on=validation
[FIXED_TEACHER_TABLE1_CUB200] h200_build=543 root=... sha256=06f75192b1c108c89e480843cb4f72dfb28aa762d7b11e7ac327333dd54b51f5
[OURS_ADAPTER] all_blocks=True common_channels=192 common_grid=14
```

## 정상 완료 확인 문구

```text
[FINAL_TEST_ONCE][OURS] selected_epoch=... best_val_top1=...% test_top1=...% evaluations=1
[FINAL_RESULT] method=OURS best_val_top1=...% final_test_top1=...% test_evaluations=1
[DONE] Table-1 CUB student task completed successfully.
```

## 완료 판정

```text
300 epochs가 완료되고 summary.json의 student_key=cvt_13, selection_protocol=val_then_test_once, selection_metric=validation_top1, test_evaluations=1이어야 한다. validation split과 고정 Teacher SHA-256 계약을 모두 통과해야 한다. ours_adapter.all_blocks=true여야 한다.
```

## 첨부 파일

```text
/app/output/cub200_cvt_13_ours_b128_valselect_fixed36p4_seed1/cub200_cvt_13_ours_b128_valselect_fixed_teacher36p4_300ep_seed1_split2027/summary.json
/app/output/cub200_cvt_13_ours_b128_valselect_fixed36p4_seed1/cub200_cvt_13_ours_b128_valselect_fixed_teacher36p4_300ep_seed1_split2027/validation_split.json
/app/output/cub200_cvt_13_ours_b128_valselect_fixed36p4_seed1/cub200_cvt_13_ours_b128_valselect_fixed_teacher36p4_300ep_seed1_split2027/student_best.pt
```
