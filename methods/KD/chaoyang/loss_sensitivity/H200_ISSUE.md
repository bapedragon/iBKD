# H200 issue: Chaoyang KD B/C timing run

## Title

```text
[Request]: 박철현 Chaoyang KD B/C loss timing run
```

## User ID

Use the account that will own the H200 job:

```text
bapedragon
```

or

```text
kau-aimslab
```

## GitHub repository

```text
https://github.com/bapedragon/IBAM_KD_H200_V2.git
```

## Command

```text
python methods/KD/chaoyang/loss_sensitivity/run.py --timing-run --data-dir /app/data/chaoyang --num-workers 4
```

## Image

```text
pytorch/pytorch:latest
```

## Language

```text
Python
```

## GPU allocation

```text
7
```

The timing runner executes B and C for two full-data epochs each and reports
their independent runtime estimates. It does not overwrite the completed A
result or any existing method result.
