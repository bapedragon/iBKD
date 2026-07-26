# CUB-200 Table-1 DeiT-Ti Vanilla-b128 full Issue

## 제목

```text
[Request]: 박철현 CUB-200 Table1 DeiT-Ti Vanilla-b128 300ep full training v2
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
python methods/table1_cub200/train.py --full-run --student deit_ti --method vanilla --batch-size 128 --data-dir /app/output/table1_cub200_deit_vanilla_full_seed1_v2/data/cub200 --output-dir /app/output/table1_cub200_deit_vanilla_full_seed1_v2 --run-name table1_cub200_deit_ti_vanilla_b128_full_300ep_seed1 --num-workers 4
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
