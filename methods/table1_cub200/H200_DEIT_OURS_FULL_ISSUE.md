# CUB-200 Table-1 DeiT-Ti Ours-b64/Ours-b128 full Issue

각 항목은 H200 Issue 입력란마다 각각 따로 복사한다.

## 제목

```text
[Request]: 박철현 CUB-200 Table1 DeiT-Ti Ours-b64/Ours-b128 2-task full training
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
python methods/table1_cub200/run_deit_ours.py --full-run --data-dir /app/output/table1_cub200_deit_ours_full_seed1/data/cub200 --output-dir /app/output/table1_cub200_deit_ours_full_seed1 --num-workers 4
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
CUB-200-2011 Table 1 확장 실험에서 완료된 Build 543 Teacher와 DeiT-Ti LG/ALG 학습에 이어 DeiT-Ti Ours를 다음 순서로 본 학습한다.

1. Ours-b64: random initialization DeiT-Ti, 입력 224×224, batch size 64, 300 epochs
2. Ours-b128: random initialization DeiT-Ti, 입력 224×224, batch size 128, 300 epochs

Teacher는 재학습하지 않는다. 두 Ours Student 모두 H200 Build 543에서 완료된 동일한 random initialization CIFAR-style ResNet-56 Teacher를 사용한다. Teacher 입력은 32×32이고, 300 epochs 학습의 best checkpoint는 epoch 275, Top-1 36.40%다.

고정 Teacher checkpoint 경로는 teachers/checkpoints/cub200_table1_resnet56_32/teacher_resnet56_cub200_32_best.pt이고 SHA-256은 06f75192b1c108c89e480843cb4f72dfb28aa762d7b11e7ac327333dd54b51f5다. 실행 전 실제 파일 hash와 manifest를 검사하고, 각 Student 학습 후 summary 및 checkpoint에 기록된 Teacher identity가 동일한지 다시 검사한다. 다른 Teacher가 제공되면 실행을 중단한다.

Student는 ImageNet pretrained weight를 사용하지 않으며 scratch로 시작한다. Ours의 기존 프로토콜과 구현을 변경하지 않는다. Ours all-block aggregation을 사용하며 DeiT-Ti의 12개 Transformer block feature는 192×14×14 공통 형태로 입력된다. optimizer는 Ours의 단일 AdamW parameter group과 전체 weight decay 0.05 조건을 유지한다. seed 1, FP32 조건을 사용한다.

타이밍 런 기준 예상 총시간은 약 2시간 24분으로 600분 제한 이내다. 마지막 로그에는 이전에 완료된 Teacher, LG, ALG와 이번 Ours-b64, Ours-b128의 Best Top-1을 한 줄로 모두 다시 출력하고 final_summary.json 및 sequence_status.json을 저장한다.
```

## 정상 시작 확인 문구

```text
[PROTOCOL_LOCK_TABLE1_CUB200_DEIT_OURS] teacher=ResNet56 scratch input=32 build=543 fixed=True student=DeiT-Ti scratch input=224 method=Ours batches=64,128 epochs=300 seed=1 fp32=True
[FIXED_TEACHER_CHECK_TABLE1_CUB200] status=PASS build=543 epoch=275 top1=36.40% input=32 sha256=06f75192b1c108c89e480843cb4f72dfb28aa762d7b11e7ac327333dd54b51f5
[PREVIOUS_RESULT_CHECK_TABLE1_CUB200_DEIT] status=PASS LG=44.51% ALG=47.70% teacher_sha256=06f75192b1c108c89e480843cb4f72dfb28aa762d7b11e7ac327333dd54b51f5
[TASK_COUNT_TABLE1_CUB200_DEIT_OURS] total=2 order=OursB64,OursB128
[POD_LIMIT_CHECK_TABLE1_CUB200_DEIT_OURS] limit=600m estimated=2h 23m 56s status=PASS
```

## 정상 완료 확인 문구

```text
[STUDENT_IDENTITY_CHECK_TABLE1_CUB200][OursB64] status=PASS student=DeiT-Ti pretrained=False input=224 batch=64 teacher_input=32 teacher_sha256=06f75192b1c108c89e480843cb4f72dfb28aa762d7b11e7ac327333dd54b51f5
[STUDENT_IDENTITY_CHECK_TABLE1_CUB200][OursB128] status=PASS student=DeiT-Ti pretrained=False input=224 batch=128 teacher_input=32 teacher_sha256=06f75192b1c108c89e480843cb4f72dfb28aa762d7b11e7ac327333dd54b51f5
[FINAL_TOP1_SUMMARY_TABLE1_CUB200_DEIT] Teacher=36.40% LG=44.51% ALG=47.70% OursB64=...% OursB128=...%
[SEQUENCE_DONE_TABLE1_CUB200_DEIT_OURS] completed_tasks=2/2
```

## 완료 판정

```text
Ours-b64와 Ours-b128이 각각 300 epochs 완료되고 completed_tasks=2/2가 출력되어야 한다. 두 identity check에서 scratch DeiT-Ti 224×224와 고정 Build 543 ResNet56-32 Teacher의 동일 SHA-256이 확인되어야 한다. 마지막 FINAL_TOP1_SUMMARY_TABLE1_CUB200_DEIT에 Teacher, LG, ALG, Ours-b64, Ours-b128의 Best Top-1 값이 모두 표시되어야 한다.
```

## 결과로 첨부할 파일 1

```text
/app/output/table1_cub200_deit_ours_full_seed1/final_summary.json
```

## 결과로 첨부할 파일 2

```text
/app/output/table1_cub200_deit_ours_full_seed1/sequence_status.json
```

## 결과로 첨부할 파일 3

```text
/app/output/table1_cub200_deit_ours_full_seed1/students/table1_cub200_deit_ti_ours_b64_full_300ep_seed1/summary.json
```

## 결과로 첨부할 파일 4

```text
/app/output/table1_cub200_deit_ours_full_seed1/students/table1_cub200_deit_ti_ours_b128_full_300ep_seed1/summary.json
```
