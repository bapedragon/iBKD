# Ours V1 learnable-aggregation alpha extraction retry Issue

각 항목은 H200 Issue 입력란마다 각각 따로 복사한다.

## 제목

```text
[Request]: 박철현 Ours V1 learnable aggregation alpha extraction retry
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
python methods/Ours/aggregation_alpha/extract_alpha.py --output-dir /app/output/ours_v1_aggregation_alpha
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
1
```

본 작업은 CPU에서 checkpoint tensor만 읽으며 CUDA를 사용하지 않는다.
Issue 양식상 필요한 최소 MIG만 요청한다.

## 요청 내용

```text
Build 564는 GitHub 원격 저장소에 추출 스크립트가 반영되기 전에 제출되어, Python 실행 전 extract_alpha.py 파일을 찾지 못하고 종료되었다. 학습, checkpoint 로딩 및 alpha 계산은 시작되지 않았으므로 해당 Build의 연구 결과는 없다. 스크립트가 반영된 원격 main에서 동일 작업을 재실행한다.

추가 학습이나 ablation 없이, 저장소에서 현재 선택된 Ours V1 DeiT-Ti 재현 best checkpoint 세 개에서 learnable aggregation의 softmax-normalized alpha 값만 추출한다. 이 결과는 아래에 명시한 checkpoint의 분석값이며, 별도의 원 논문 checkpoint에서 추출한 값으로 표기하지 않는다.

대상은 다음 세 실행으로 고정한다.

1. CIFAR-100: results/Ours/cifar100/researcher_sync_v1_300ep_seed1/student_best.pt
2. Flowers-102: results/Ours/flowers102/researcher_sync_v1_300ep_seed1/student_best.pt
3. Chaoyang: results/Ours/chaoyang/cifar100_locked_b64_v1_300ep_seed1/student_best.pt

각 checkpoint의 checkpoint["ours"]["aggregation.weights"]에 마지막 차원 softmax를 적용하여 teacher stage 3개 × student Transformer block 12개의 alpha 행렬을 얻는다. 결과 표기는 teacher stage와 student block 모두 1-based indexing을 사용한다.

각 checkpoint에서 method=Ours, student=deit_ti, dataset identity와 aggregation.weights shape=3x12를 검증한다. 각 alpha가 유한한 비음수이고 stage별 합이 1인지 확인한다. 또한 계산된 전체 3x12 행렬을 대응하는 run_summary.json의 aggregation_weights와 절대오차 1e-6 이내로 교차 검증한다.

결과에는 전체 alpha 108개, stage별 Top-3 student block, best-checkpoint epoch/Top-1/SHA-256을 기록한다. 새 학습, 데이터셋 로딩, teacher/student forward, Ours V2, historical run, batch/lambda/Table-4 control, CUB-200 실험은 수행하지 않는다.
```

## 정상 시작 확인 문구

```text
[OURS_V1_ALPHA_PROTOCOL] method=OursV1 datasets=cifar100,flowers102,chaoyang source=student_best.pt normalization=softmax shape=3x12 device=cpu training=False
```

## 정상 완료 확인 문구

```text
[ALPHA_CHECK][cifar100] status=PASS ... shape=3x12 row_sums=1 ...
[ALPHA_CHECK][flowers102] status=PASS ... shape=3x12 row_sums=1 ...
[ALPHA_CHECK][chaoyang] status=PASS ... shape=3x12 row_sums=1 ...
[ALPHA_EXTRACTION_DONE] status=PASS datasets=3 matrices=3 values=108 training=False ours_v2=False
```

## 완료 판정

```text
세 데이터셋의 ALPHA_CHECK가 모두 PASS이고, checkpoint에서 계산한 softmax alpha와 각 run_summary.json의 aggregation_weights 차이가 1e-6 이하여야 한다. 각 stage alpha 합은 1이어야 하며, 최종 ALPHA_EXTRACTION_DONE에 datasets=3, matrices=3, values=108, training=False, ours_v2=False가 출력되어야 한다. Python traceback, checkpoint identity mismatch, shape mismatch, non-finite alpha 또는 summary cross-check 실패가 있으면 완료로 판정하지 않는다.
```

## 결과로 첨부할 파일 1

```text
/app/output/ours_v1_aggregation_alpha/aggregation_alpha.json
```

## 결과로 첨부할 파일 2

```text
/app/output/ours_v1_aggregation_alpha/aggregation_alpha.csv
```

## 결과로 첨부할 파일 3

```text
/app/output/ours_v1_aggregation_alpha/aggregation_alpha.md
```
