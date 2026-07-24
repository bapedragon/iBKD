# CUB-200 ResNet50-224 H200 Issues

This is the separate **224 x 224 transfer-learning family**. Do not mount,
reuse, or compare it as the existing ResNet56-32 scratch family. Run the timing
Issue first.

## 1. Timing Issue

### 제목

```text
[Request]: 박철현 CUB-200 ImageNet-pretrained ResNet50-224 teacher + LG224 + ALG224 + Ours224 timing run
```

### 개인 계정 사용자 ID

```text
bapedragon
```

### 연구실 계정 사용자 ID

```text
kau-aimslab
```

### 제출 계정

```text
bapedragon 또는 kau-aimslab 중 실제 제출하는 계정
```

### 실행할 코드의 GitHub 링크

```text
https://github.com/bapedragon/IBAM_KD_H200_V2.git
```

### 코드 실행 명령어

```text
python methods/run_cub200_resnet50_224_lg_alg_ours.py --timing-run --num-workers 4 --output-dir /app/output/cub200_resnet50_224_lg_alg_ours_timing_seed1
```

### 사용할 이미지

```text
pytorch/pytorch:latest
```

### 사용 언어

```text
Python
```

### GPU 할당량 (MIG 개수)

```text
7
```

통과 로그:

```text
[PROTOCOL_FAMILY] cub200_common_transfer_resnet50_224 separate_from=cub200_resnet56_32_scratch
[RESOLUTION_LOCK] teacher_input=224 student_input=224 no_32px_teacher=True
[FEATURE_CONTRACT] teacher=ResNet50 stages=(layer2,layer3,layer4) channels=(512,1024,2048) grids=(28,14,7)
[SEQUENCE_DONE_224] completed_tasks=4/4
[POD_LIMIT_CHECK_224] status=PASS
[FINAL_TOP1_SUMMARY_224] Teacher224=...% LG224=...% ALG224=...% Ours224=...%
```

`FINAL_TOP1_SUMMARY_224`가 마지막 줄이므로 긴 로그가 잘려도 네 결과를
한 번에 확인할 수 있다. 타이밍 run은 실제 H200에서 teacher/LG/ALG/Ours
각각 2 epoch의 forward/backward를 수행하므로 224 입력의 shape 불일치와
CUDA OOM을 확인할 수 있다. 최종 정확도와 수렴 여부는 판정하지 않는다.

## 2. Full Issue

Timing Issue가 `completed_tasks=4/4`, `POD_LIMIT_CHECK_224 status=PASS`로
끝난 뒤에만 제출한다.

### 제목

```text
[Request]: 박철현 CUB-200 ImageNet-pretrained ResNet50-224 teacher 200ep + LG224 + ALG224 + Ours224 300ep training
```

### 개인 계정 사용자 ID

```text
bapedragon
```

### 연구실 계정 사용자 ID

```text
kau-aimslab
```

### 제출 계정

```text
bapedragon 또는 kau-aimslab 중 실제 제출하는 계정
```

### 실행할 코드의 GitHub 링크

```text
https://github.com/bapedragon/IBAM_KD_H200_V2.git
```

### 코드 실행 명령어

```text
python methods/run_cub200_resnet50_224_lg_alg_ours.py --full-run --num-workers 4 --output-dir /app/output/cub200_resnet50_224_lg_alg_ours_full_seed1
```

### 사용할 이미지

```text
pytorch/pytorch:latest
```

### 사용 언어

```text
Python
```

### GPU 할당량 (MIG 개수)

```text
7
```

Full run은 200-epoch ResNet50-224 teacher를 먼저 만들고 그 동일한
manifest/checkpoint를 LG224, ALG224, Ours224의 300-epoch 학습에 순서대로
사용한다. 2-epoch timing teacher는 full student run에 재사용되지 않는다.
