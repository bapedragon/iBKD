# CUB-200 full ImageNet-transfer: H200 full Issue

기존 300-epoch timing run에서 모델, pretrained weight, batch, shape,
forward/backward 및 메모리 경로가 모두 통과했다. 이번 변경은 모든 student
학습 horizon을 300에서 100 epochs로 줄이는 epoch-only 변경이므로 그
timing 결과는 보수적인 검증으로 유효하다. 각 항목은 H200 Issue 입력란마다
각각 따로 복사한다.

## 제목

```text
[Request]: 박철현 CUB-200 pretrained Teacher 200ep + pretrained Vanilla-b128 + LG + ALG + Ours-b64/b128 100ep full training
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
python methods/run_cub200_full_transfer_all.py --full-run --num-workers 4 --output-dir /app/output/cub200_full_transfer_all_100ep_full_seed1
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
[PRETRAINING_LOCK] teacher_pretrained=True all_students_pretrained=True teacher_source=torchvision.ResNet50_Weights.IMAGENET1K_V2 student_source=timm/deit_tiny_patch16_224.fb_in1k
[EPOCH_LOCK] teacher_planned_epochs=200 all_students_planned_epochs=100
[RESOLUTION_LOCK] teacher_input=224 all_student_inputs=224 no_32px_teacher=True
[TEACHER_IDENTITY_CHECK] status=PASS pretrained=True source=torchvision.ResNet50_Weights.IMAGENET1K_V2 input=224
[SEQUENCE_DONE_224_FULL_TRANSFER] completed_tasks=6/6
[POD_LIMIT_CHECK_224_FULL_TRANSFER] status=PASS
[FINAL_TOP1_SUMMARY_224_FULL_TRANSFER] Teacher=...% VanillaB128=...% LG=...% ALG=...% OursB64=...% OursB128=...%
```

## 본학습 범위

```text
ResNet50 teacher는 torchvision ImageNet1K-V2 pretrained weight에서 시작하여 CUB-200 224×224로 200 epochs fine-tuning한다. Vanilla, LG, ALG, Ours student는 모두 timm DeiT-Ti ImageNet-1K pretrained backbone에서 시작하여 CUB-200 224×224로 각각 100 epochs 학습한다. Vanilla는 batch 128만 실행하고 LG와 ALG는 공식 LG 기반의 고정 batch 128을 유지한다. Ours는 현재 기본 batch 64와 batch-only 비교인 128을 모두 실행한다. 기존 300-epoch full-transfer 출력과 혼동되지 않도록 모든 student run name에 100ep가 기록된다.
```

## 완료 판정

```text
Teacher와 다섯 student 작업이 모두 완료되어 completed_tasks=6/6이 출력되고 마지막 FINAL_TOP1_SUMMARY_224_FULL_TRANSFER에 Teacher, VanillaB128, LG, ALG, OursB64, OursB128의 Best Top-1 값이 모두 표시되면 완료이다.
```
