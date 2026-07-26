# CUB-200 both-scratch 300-epoch students: H200 timing Issue

각 항목은 H200 Issue 입력란마다 각각 따로 복사한다.

## 제목

```text
[Request]: 박철현 CUB-200 scratch Teacher 200ep + scratch Vanilla/LG/ALG/Ours 300ep timing run
```

## 개인 계정 사용자 ID

```text
bapedragon
```

## 연구실 계정 사용자 ID

```text
kau-aimslab
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
python methods/run_cub200_both_scratch_300ep_all.py --timing-run --num-workers 4 --output-dir /app/output/cub200_both_scratch_300ep_timing_seed1
```

## 사용할 이미지

```text
pytorch/pytorch:latest
```

## 사용 언어

```text
Python
```

## GPU 할당량 (MIG 개수)

```text
7
```

## 확인할 마지막 로그

```text
[CONTROLLED_HORIZON] reference=cub200_resnet50_deit_ti_224_both_scratch_100ep only_changed_factor=all_student_epochs_100_to_300 teacher_planned_epochs_unchanged=200
[PRETRAINING_LOCK] teacher_pretrained=False all_students_pretrained=False teacher_source=none student_source=none
[EPOCH_LOCK] teacher_planned_epochs=200 all_students_planned_epochs=300
[RESOLUTION_LOCK] teacher_input=224 all_student_inputs=224 no_32px_teacher=True
[SEQUENCE_DONE_224_BOTH_SCRATCH_300EP] completed_tasks=6/6
[POD_LIMIT_CHECK_224_BOTH_SCRATCH_300EP] status=PASS
[FINAL_TOP1_SUMMARY_224_BOTH_SCRATCH_300EP] Teacher=...% VanillaB128=...% LG=...% ALG=...% OursB64=...% OursB128=...%
```

## 실행 범위

```text
완료된 both-scratch 100-epoch 실험과 Teacher/Student architecture, random initialization, CUB-200 split, 224×224 입력, augmentation, optimizer, seed, method 설정, 실행 순서와 batch를 동일하게 유지한다. Teacher는 기존과 동일하게 200 epochs로 학습하고 고정한다. VanillaB128, LG, ALG, OursB64, OursB128의 student 학습 horizon만 모두 100에서 300 epochs로 늘린다. Timing mode에서는 각 작업을 실제 2 epochs 실행해 shape, forward/backward, 메모리 및 600분 내 총 예상시간을 확인한다.
```

## 완료 판정

```text
Teacher와 다섯 student 작업이 모두 완료되어 completed_tasks=6/6이 출력되고 EPOCH_LOCK이 teacher=200 및 all_students=300이며 POD_LIMIT_CHECK가 PASS이면 timing run 완료이다.
```
