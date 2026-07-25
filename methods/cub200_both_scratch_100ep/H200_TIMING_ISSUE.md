# CUB-200 both-scratch paired control: H200 timing Issue

각 항목은 H200 Issue 입력란마다 각각 따로 복사한다.

## 제목

```text
[Request]: 박철현 CUB-200 Teacher+Student 모두 scratch 224×224 paired 100ep timing run
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
python methods/run_cub200_both_scratch_100ep_all.py --timing-run --num-workers 4 --output-dir /app/output/cub200_both_scratch_100ep_timing_seed1
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
[CONTROLLED_PAIR] reference=cub200_resnet50_deit_ti_224_both_imagenet_pretrained only_changed_factor=teacher_and_student_initialization
[PRETRAINING_LOCK] teacher_pretrained=False all_students_pretrained=False teacher_source=none student_source=none
[EPOCH_LOCK] teacher_planned_epochs=200 all_students_planned_epochs=100
[RESOLUTION_LOCK] teacher_input=224 all_student_inputs=224 no_32px_teacher=True
[SEQUENCE_DONE_224_BOTH_SCRATCH_100EP] completed_tasks=6/6
[POD_LIMIT_CHECK_224_BOTH_SCRATCH_100EP] status=PASS
[FINAL_TOP1_SUMMARY_224_BOTH_SCRATCH_100EP] Teacher=...% VanillaB128=...% LG=...% ALG=...% OursB64=...% OursB128=...%
```

## 실행 범위

```text
기존 fully pretrained 100-epoch 실험과 ResNet50/DeiT-Ti, CUB-200 split, 224×224 입력, teacher 200 epochs, student 100 epochs, 학습 순서 및 batch를 동일하게 유지한다. ImageNet initialization만 teacher와 모든 student에서 제거한다. Vanilla는 batch 128, LG와 ALG는 batch 128, Ours는 batch 64와 128을 실행한다. 각 작업은 실제 2 epochs를 수행하여 shape, forward/backward, 메모리와 총 예상 시간을 검증한다.
```

## 완료 판정

```text
Teacher와 다섯 student 작업이 모두 완료되어 completed_tasks=6/6이 출력되고 PRETRAINING_LOCK이 teacher_pretrained=False 및 all_students_pretrained=False이며 POD_LIMIT_CHECK가 PASS이면 timing run 완료이다.
```
