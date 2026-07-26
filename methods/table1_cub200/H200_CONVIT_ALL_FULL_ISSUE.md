# CUB-200 Table-1 ConViT-Ti all-five full Issue

## 제목

```text
[Request]: 박철현 CUB-200 Table1 ConViT-Ti Vanilla/LG/ALG/Ours64/Ours128 300ep full training
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
python methods/table1_cub200/run_backbone_all.py --full-run --student convit_ti --data-dir /app/output/table1_cub200_convit_all_full_seed1/data/cub200 --output-dir /app/output/table1_cub200_convit_all_full_seed1 --num-workers 4
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
