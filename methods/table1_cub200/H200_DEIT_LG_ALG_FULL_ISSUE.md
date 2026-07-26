# CUB-200 Table-1 ResNet56-32 Teacher + DeiT-Ti LG/ALG full Issue

> 완료 기록: H200 build 543에서 Teacher 36.40%, LG 44.51%, ALG 47.70%로
> 3/3 완료했다. 이 문서는 최초 teacher 생성 실행의 provenance로
> 보존한다. 이후 Table-1 student 학습은 이 명령으로 teacher를 다시
> 만들지 않고 저장소의 고정 build-543 teacher를 사용한다.

각 항목은 H200 Issue 입력란마다 각각 따로 복사한다.

## 제목

```text
[Request]: 박철현 CUB-200 Table1 ResNet56-32 Teacher + DeiT-Ti LG/ALG 3-task full training
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
python methods/table1_cub200/run_deit_lg_alg.py --full-run --data-dir /app/output/table1_cub200_deit_lg_alg_full_seed1/data/cub200 --output-dir /app/output/table1_cub200_deit_lg_alg_full_seed1 --num-workers 4
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
CUB-200-2011 공식 train/test split에서 Table 1 확장 실험의 첫 세 작업을 순서대로 본 학습한다.

1. Teacher: random initialization CIFAR-style ResNet-56, 입력 32×32, batch size 128, 300 epochs
2. LG: random initialization DeiT-Ti, 입력 224×224, 공식 LG feature guidance, batch size 128, 300 epochs
3. ALG: random initialization DeiT-Ti, 입력 224×224, 공식 LG feature loss와 ALG 논문의 controller 수식, batch size 128, 300 epochs

Teacher를 이 실행에서 새로 학습하며, LG와 ALG는 새 Teacher 학습에서 생성된 동일한 best checkpoint와 SHA-256 hash를 공유한다. 저장소에 기존 보관된 Teacher checkpoint를 재사용하지 않는다.

LG는 공개된 공식 LG 코드의 DeiT-Ti 백본, student block [0,6,11], teacher stage [0,1,2], stage별 1×1 projection, 더 큰 grid로의 bilinear resize, stage별 mean MSE 합 및 beta 2.5를 사용한다.

ALG는 공식 공개 구현이 없으므로 공식 LG 코드 기반 feature loss에 ALG 논문에서 명시한 controller equations, smoothing window 50, threshold -0.02, >= stop boundary, controller warm-up 0을 적용한다.

모든 모델은 random initialization이며 ImageNet pretrained weight를 사용하지 않는다. Teacher 입력만 32×32이고 두 DeiT-Ti Student 입력은 224×224이다. seed 1, FP32 조건을 사용한다.

타이밍 런 측정 기준 예상 총시간은 약 2시간 54분으로 600분 제한 이내다. 마지막 로그에 Teacher, LG, ALG의 Best Top-1을 한 줄로 다시 출력하고 final_summary.json 및 sequence_status.json을 저장한다.
```

## 정상 시작 확인 문구

```text
[PROTOCOL_LOCK_TABLE1_CUB200_DEIT_LG_ALG] teacher=ResNet56 scratch input=32 batch=128 epochs=300 students=DeiT-Ti scratch input=224 methods=LG,ALG batch=128 epochs=300 seed=1 fp32=True
[TASK_COUNT_TABLE1_CUB200_DEIT_LG_ALG] total=3 order=Teacher,LG,ALG
[POD_LIMIT_CHECK_TABLE1_CUB200_DEIT_LG_ALG] limit=600m estimated=2h 53m 58s status=PASS
```

## 정상 완료 확인 문구

```text
[TEACHER_IDENTITY_CHECK_TABLE1_CUB200] status=PASS model=ResNet56 pretrained=False input=32 sha256=...
[STUDENT_IDENTITY_CHECK_TABLE1_CUB200][LG] status=PASS student=DeiT-Ti pretrained=False input=224 batch=128 teacher_input=32
[STUDENT_IDENTITY_CHECK_TABLE1_CUB200][ALG] status=PASS student=DeiT-Ti pretrained=False input=224 batch=128 teacher_input=32
[FINAL_TOP1_SUMMARY_TABLE1_CUB200_DEIT_LG_ALG] Teacher=...% LG=...% ALG=...%
[SEQUENCE_DONE_TABLE1_CUB200_DEIT_LG_ALG] completed_tasks=3/3
```

## 완료 판정

```text
Teacher, LG, ALG 세 작업이 모두 300 epochs 완료되고 completed_tasks=3/3이 출력되어야 한다. 마지막 FINAL_TOP1_SUMMARY_TABLE1_CUB200_DEIT_LG_ALG에 Teacher, LG, ALG의 Best Top-1 값이 모두 표시되어야 한다. LG와 ALG의 student identity check에서 student input=224, teacher_input=32가 확인되어야 한다.
```

## 결과로 첨부할 파일 1

```text
/app/output/table1_cub200_deit_lg_alg_full_seed1/final_summary.json
```

## 결과로 첨부할 파일 2

```text
/app/output/table1_cub200_deit_lg_alg_full_seed1/sequence_status.json
```

## 결과로 첨부할 파일 3

```text
/app/output/table1_cub200_deit_lg_alg_full_seed1/teacher/table1_cub200_teacher_resnet56_32_b128_full_300ep_seed1/summary.json
```

## 결과로 첨부할 파일 4

```text
/app/output/table1_cub200_deit_lg_alg_full_seed1/students/table1_cub200_deit_ti_lg_b128_full_300ep_seed1/summary.json
```

## 결과로 첨부할 파일 5

```text
/app/output/table1_cub200_deit_lg_alg_full_seed1/students/table1_cub200_deit_ti_alg_b128_full_300ep_seed1/summary.json
```
