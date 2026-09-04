# Flowers-102 + Chaoyang Ours V1 lambda=0.25 full-run Issue

각 항목은 H200 Issue 입력란마다 각각 따로 복사한다.

## 제목

```text
[Request]: 박철현 Ours V1 Flowers-102/Chaoyang lambda 0.25 2-task 300-epoch full training
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
https://github.com/bapedragon/iBKD.git
```

## 코드 실행 명령어

```text
python methods/Ours/lambda_transfer/run_flowers_chaoyang.py --full-run --num-workers 4 --output-dir /app/output/ours_v1_lambda_0p25_flowers_chaoyang_300ep_seed1
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

## 요청 내용

```text
CIFAR-100 Table 7에서 Ours V1 lambda=0.25가 lambda=0.5보다 +0.50%p 높게 나온 결과가 다른 데이터셋에도 전이되는지 확인한다. Flowers-102와 Chaoyang에서 기존의 완전 보관된 lambda=0.5 실행을 대조군으로 재사용하고, 다른 모든 조건을 동일하게 고정한 채 lambda만 0.25로 변경하여 각각 300 epochs 학습한다.

실행 순서는 다음과 같다.

1. Flowers-102 Ours V1: scratch DeiT-Ti, input 224, train/eval batch 64/200, 300 epochs, seed 1, FP32, lambda=0.25
2. Chaoyang Ours V1: scratch DeiT-Ti, input 224, train/eval batch 64/200, 300 epochs, seed 1, FP32, lambda=0.25

loss는 CE + beta(e) * [lambda * L_fuse + (1-lambda) * L_align]이고, 이번 실행의 feature loss는 0.25 * L_fuse + 0.75 * L_align이다. lambda 외에는 teacher checkpoint, student initialization, 데이터 split, augmentation, optimizer AdamW, LR 5e-4 -> 5e-6 cosine, weight decay 0.05, LR warm-up 20, ALG controller beta 2.5 / tau -0.02 / smoothing 50 / controller warm-up 20, larger-grid 32/16/14, Ours V1 모듈 및 seed를 변경하지 않는다.

Flowers-102의 lambda=0.5 대조 결과는 results/Ours/flowers102/researcher_sync_v1_300ep_seed1이며 best Top-1 74.81%, last 74.21%, guidance stop epoch 182다. Chaoyang의 lambda=0.5 대조 결과는 results/Ours/chaoyang/cifar100_locked_b64_v1_300ep_seed1이며 best Top-1 81.11%, last 80.22%, guidance stop epoch 192다. Chaoyang 81.95% 값은 개별 checkpoint와 summary가 저장소에 누락된 로그 전용 값이므로 이번 직접 비교 기준에서 제외한다.

실행기는 학습 전 두 대조 summary의 lambda, 전체 고정 인자, teacher SHA-256 및 Ours source SHA-256을 검사한다. 각 새 실행이 끝난 뒤 동일 항목과 active lambda=0.25를 다시 검사하고, 기존 lambda=0.5 대비 Best Top-1 차이를 sequence_summary.json에 기록한다.

Flowers-102 데이터는 없으면 공식 파일을 다운로드하고 checksum을 검증한다. Chaoyang은 /app/data/chaoyang에 마운트된 공식 4,021/2,139 train/test 데이터를 사용하며 전체 JSON record, class count 및 이미지 파일을 검증한다. 두 lambda=0.5 실행에서 측정된 총 소요 시간은 약 1시간 36분이므로 600분 Pod 제한 이내다.

마지막 로그 줄에는 Flowers-102와 Chaoyang의 lambda=0.25 Best Top-1 두 값을 반드시 함께 출력한다.
```

## 정상 시작 확인 문구

```text
[SEQUENCE] Flowers-102 -> Chaoyang
[PROTOCOL_LOCK] only_change=lambda reference_lambda=0.5 active_lambda=0.25
[REFERENCE_CHECK][Flowers-102] status=PASS lambda=0.5 best_top1=74.81% ...
[REFERENCE_CHECK][Chaoyang] status=PASS lambda=0.5 best_top1=81.11% ...
```

## 정상 완료 확인 문구

```text
[FINAL_RESULT][Flowers-102][lambda=0.25] best_top1=...% latest_top1=...% reference_lambda_0p5=74.81% delta=...pp
[FINAL_RESULT][Chaoyang][lambda=0.25] best_top1=...% latest_top1=...% reference_lambda_0p5=81.11% delta=...pp
[POD_LIMIT_CHECK] status=PASS limit=10h estimated=...
[SEQUENCE_DONE] completed_tasks=2/2
[DONE] Flowers-102 and Chaoyang lambda=0.25 tasks completed.
[FINAL_TOP1_SUMMARY_LAMBDA_0P25] Flowers102=...% Chaoyang=...%
```

## 완료 판정

```text
Flowers-102와 Chaoyang이 각각 300 epochs를 완료하고 두 결과 summary의 fusion_ratio가 0.25여야 한다. 두 실행 모두 기존 lambda=0.5 대조군과 teacher SHA-256 및 Ours source SHA-256이 일치하고, 고정 프로토콜 항목 검사를 모두 통과해야 한다. sequence_status.json과 sequence_summary.json의 status가 complete이고 completed task가 2개여야 한다. 마지막 FINAL_TOP1_SUMMARY_LAMBDA_0P25 줄에 Flowers102와 Chaoyang의 Best Top-1 두 값이 모두 출력되어야 한다. Python traceback, dataset/teacher/reference audit 실패 또는 한 작업 누락이 있으면 완료로 판정하지 않는다.
```

## 결과로 첨부할 파일 1

```text
/app/output/ours_v1_lambda_0p25_flowers_chaoyang_300ep_seed1/sequence_summary.json
```

## 결과로 첨부할 파일 2

```text
/app/output/ours_v1_lambda_0p25_flowers_chaoyang_300ep_seed1/sequence_status.json
```

## 결과로 첨부할 파일 3

```text
/app/output/ours_v1_lambda_0p25_flowers_chaoyang_300ep_seed1/flowers102/ours_v1_flowers102_lambda_0p25_b64_300ep_seed1/summary.json
```

## 결과로 첨부할 파일 4

```text
/app/output/ours_v1_lambda_0p25_flowers_chaoyang_300ep_seed1/flowers102/ours_v1_flowers102_lambda_0p25_b64_300ep_seed1/student_best.pt
```

## 결과로 첨부할 파일 5

```text
/app/output/ours_v1_lambda_0p25_flowers_chaoyang_300ep_seed1/chaoyang/ours_v1_chaoyang_lambda_0p25_b64_300ep_seed1/summary.json
```

## 결과로 첨부할 파일 6

```text
/app/output/ours_v1_lambda_0p25_flowers_chaoyang_300ep_seed1/chaoyang/ours_v1_chaoyang_lambda_0p25_b64_300ep_seed1/student_best.pt
```
