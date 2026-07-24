# CUB-200 ResNet50-224 scratch family: H200 full Issue

이 본학습 Issue는 같은 commit의 timing run이
`completed_tasks=6/6`, `POD_LIMIT_CHECK_224_SCRATCH status=PASS`로 끝났다는
전제에서 제출한다. 각 항목은 H200 Issue 입력란에 따로 복사한다.

## 제목

```text
[Request]: 박철현 CUB-200 random-init ResNet50-224 teacher 200ep + Vanilla-b128 + LG + ALG + Vanilla-b64 + Ours 300ep full training
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
python methods/run_cub200_resnet50_224_scratch_all.py --full-run --num-workers 4 --output-dir /app/output/cub200_resnet50_224_scratch_all_full_seed1
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
[PRETRAINING_LOCK] teacher_pretrained=False all_students_pretrained=False
[RESOLUTION_LOCK] teacher_input=224 student_input=224 no_32px_teacher=True
[TEACHER_IDENTITY_CHECK] status=PASS pretrained=False model=resnet50_cub200_scratch_224 input=224
[SEQUENCE_DONE_224_SCRATCH] completed_tasks=6/6
[POD_LIMIT_CHECK_224_SCRATCH] status=PASS
[FINAL_TOP1_SUMMARY_224_SCRATCH] TeacherScratch224=...% VanillaB128=...% LG=...% ALG=...% VanillaB64=...% Ours=...%
```

## 본학습 범위

```text
ResNet50 teacher는 random initialization, 224×224, 200 epochs로 새로 학습한다. 이어서 DeiT-Ti scratch 224×224를 Vanilla-b128, LG, ALG, Vanilla-b64, Ours 순서로 각각 300 epochs 학습한다. 모든 결과는 cub200_resnet50_224_scratch 전용 출력에 저장하며 기존 ResNet56-32 및 ImageNet-pretrained ResNet50-224 결과를 재사용하거나 덮어쓰지 않는다.
```

## 완료 판정

```text
Teacher와 다섯 student 작업이 모두 완료되어 completed_tasks=6/6이 출력되고, 마지막 FINAL_TOP1_SUMMARY_224_SCRATCH에 여섯 Best Top-1 값이 모두 표시되면 완료이다.
```
