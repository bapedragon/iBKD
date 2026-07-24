# CUB-200 ResNet50-224 scratch family: H200 timing Issue

아래 항목은 H200 timing Issue에 각각 따로 복사한다. 기존
`ResNet56-32 scratch` 및 `ResNet50-224 ImageNet-pretrained` 결과와
출력 폴더를 공유하지 않는다.

## 제목

```text
[Request]: 박철현 CUB-200 random-init ResNet50-224 teacher + Vanilla-b128 + LG + ALG + Vanilla-b64 + Ours timing run
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
python methods/run_cub200_resnet50_224_scratch_all.py --timing-run --num-workers 4 --output-dir /app/output/cub200_resnet50_224_scratch_all_timing_seed1
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

## Timing run 판정 기준

```text
Teacher, Vanilla-b128, LG, ALG, Vanilla-b64, Ours가 각각 실제 2 epoch를 완료하고 completed_tasks=6/6 및 POD_LIMIT_CHECK_224_SCRATCH status=PASS가 출력되면 통과이다. Timing 정확도는 최종 성능으로 해석하지 않으며, timing teacher 체크포인트는 본학습에 재사용하지 않는다.
```
