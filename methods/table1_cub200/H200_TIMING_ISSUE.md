# CUB-200 Table-1 seven-backbone 36-task timing Issue

Copy each field separately.

## 제목

```text
[Request]: 박철현 CUB-200 ResNet56-32 + Table1 7백본 Vanilla/LG/ALG/Ours64/Ours128 36-task timing run
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
python methods/table1_cub200/run_timing.py --timing-run --data-dir /app/output/table1_cub200_7backbone_36task_timing_seed1_v2/data/cub200 --output-dir /app/output/table1_cub200_7backbone_36task_timing_seed1_v2 --num-workers 4
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
CUB-200-2011 공식 train/test split에서 Table 1 확장 timing run을 수행한다. Teacher는 scratch CIFAR-style ResNet56, 입력 32x32, batch 128, 300 epochs 계획이다. Student는 scratch 224x224의 DeiT-Ti, ConViT-Ti, CvT-13, PiT-Ti, PVTv2-B0, T2T-ViT-7, T2T-ViT-14이다. 각 백본에서 Vanilla-b128, 공식 LG-b128, ALG 논문 controller 기반 ALG-b128, Ours-b64, Ours-b128을 측정한다. 총 작업 수는 Teacher 1 + 7백본 x 5설정 = 36개다. Timing mode는 각 작업을 전체 데이터로 실제 2 epochs 수행하고 300-epoch 예상 시간을 계산한다. 모든 guided 학생은 H200 build 543의 고정 ResNet56-32 Teacher checkpoint(36.40%, epoch 275, SHA-256 06f75192b1c108c89e480843cb4f72dfb28aa762d7b11e7ac327333dd54b51f5)를 공유하며, 2-epoch timing Teacher checkpoint는 학생 guidance에 사용하지 않는다. 다른 teacher manifest나 hash는 학습 시작 전에 거부한다. 마지막에 36개 개별 예상 시간, Teacher 포함 총 시간, 기존 Teacher 재사용 시 총 시간, 600분 제한 PASS/FAIL을 출력하고 timing_summary.json 및 timing_summary.csv를 저장한다.
```

## 정상 완료 확인 문구

```text
[TASK_COUNT] requested_previous=29 corrected_with_ours_b64_b128=36 formula=1_teacher+7_backbones*5_student_settings
[FINAL_TOTAL_ESTIMATE] tasks=36 teacher=... students35=... with_teacher=... reuse_teacher=...
[POD_LIMIT_CHECK] limit=600m estimated=...m status=PASS|FAIL
[SEQUENCE_DONE] completed_tasks=36/36
```

## 결과로 첨부할 파일

```text
/app/output/table1_cub200_7backbone_36task_timing_seed1_v2/timing_summary.json
/app/output/table1_cub200_7backbone_36task_timing_seed1_v2/timing_summary.csv
/app/output/table1_cub200_7backbone_36task_timing_seed1_v2/sequence_status.json
```
