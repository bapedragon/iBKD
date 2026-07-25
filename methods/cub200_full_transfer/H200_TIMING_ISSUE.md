# CUB-200 full ImageNet-transfer: H200 timing Issue

아래 항목은 H200 Issue 입력란마다 각각 따로 복사한다. 기존 CUB scratch 및
teacher-only-pretrained 결과와 출력 폴더를 공유하지 않는다.

## 제목

```text
[Request]: 박철현 CUB-200 pretrained Teacher + pretrained Vanilla-b128 + LG + ALG + Ours-b64/b128 timing run
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
python methods/run_cub200_full_transfer_all.py --timing-run --num-workers 4 --output-dir /app/output/cub200_full_transfer_all_timing_seed1
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
[RESOLUTION_LOCK] teacher_input=224 all_student_inputs=224 no_32px_teacher=True
[TEACHER_IDENTITY_CHECK] status=PASS pretrained=True source=torchvision.ResNet50_Weights.IMAGENET1K_V2 input=224
[STUDENT_IDENTITY_CHECK][VanillaB128] status=PASS pretrained=True input=224 batch=128
[STUDENT_IDENTITY_CHECK][LG] status=PASS pretrained=True input=224 batch=128
[STUDENT_IDENTITY_CHECK][ALG] status=PASS pretrained=True input=224 batch=128
[STUDENT_IDENTITY_CHECK][OursB64] status=PASS pretrained=True input=224 batch=64
[STUDENT_IDENTITY_CHECK][OursB128] status=PASS pretrained=True input=224 batch=128
[SEQUENCE_DONE_224_FULL_TRANSFER] completed_tasks=6/6
[POD_LIMIT_CHECK_224_FULL_TRANSFER] status=PASS
[FINAL_TOP1_SUMMARY_224_FULL_TRANSFER] Teacher=...% VanillaB128=...% LG=...% ALG=...% OursB64=...% OursB128=...%
```

## Timing run 판정 기준

```text
Teacher, Vanilla-b128, LG, ALG, Ours-b64, Ours-b128이 각각 실제 2 epochs를 완료하고 completed_tasks=6/6 및 POD_LIMIT_CHECK_224_FULL_TRANSFER status=PASS가 출력되면 통과이다. Timing 정확도는 최종 성능으로 해석하지 않으며 timing teacher 체크포인트는 본학습에 재사용하지 않는다.
```

