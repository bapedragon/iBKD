# CUB-200 both-scratch paired control: H200 full Issue

각 항목은 H200 Issue 입력란마다 각각 따로 복사한다.

## 제목

```text
[Request]: 박철현 CUB-200 scratch Teacher 200ep + scratch Vanilla-b128 + LG + ALG + Ours-b64/b128 100ep full training
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
python methods/run_cub200_both_scratch_100ep_all.py --full-run --num-workers 4 --output-dir /app/output/cub200_both_scratch_100ep_full_seed1
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
[CONTROLLED_PAIR] reference=cub200_resnet50_deit_ti_224_both_imagenet_pretrained only_changed_factor=teacher_and_student_initialization
[PRETRAINING_LOCK] teacher_pretrained=False all_students_pretrained=False teacher_source=none student_source=none
[EPOCH_LOCK] teacher_planned_epochs=200 all_students_planned_epochs=100
[RESOLUTION_LOCK] teacher_input=224 all_student_inputs=224 no_32px_teacher=True
[TEACHER_IDENTITY_CHECK] status=PASS pretrained=False source=none input=224
[SEQUENCE_DONE_224_BOTH_SCRATCH_100EP] completed_tasks=6/6
[POD_LIMIT_CHECK_224_BOTH_SCRATCH_100EP] status=PASS
[FINAL_TOP1_SUMMARY_224_BOTH_SCRATCH_100EP] Teacher=...% VanillaB128=...% LG=...% ALG=...% OursB64=...% OursB128=...%
```

## 본학습 범위

```text
ResNet50 teacher와 모든 DeiT-Ti student를 random initialization에서 시작한다. 기존 fully pretrained 실험과 CUB-200 split, 224×224 입력, augmentation, optimizer, scheduler, seed, method 설정, 실행 순서 및 batch를 동일하게 유지한다. Teacher는 200 epochs, Vanilla/LG/ALG/Ours는 각각 100 epochs 학습한다. Vanilla는 batch 128만 실행하고 LG와 ALG는 고정 batch 128, Ours는 현재 기본 batch 64와 batch-only 비교인 128을 모두 실행한다. 기존 300-epoch scratch 출력과 혼동되지 않도록 runner, output 및 모든 run name에 both_scratch_100ep가 기록된다.
```

## 완료 판정

```text
Teacher와 다섯 student 작업이 모두 완료되어 completed_tasks=6/6이 출력되고 마지막 FINAL_TOP1_SUMMARY_224_BOTH_SCRATCH_100EP에 Teacher, VanillaB128, LG, ALG, OursB64, OursB128의 Best Top-1 값이 모두 표시되면 완료이다.
```
