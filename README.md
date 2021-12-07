# 설치
## 환경
- ubuntu18.04  
- cudatoolkit 10.2

## conda로 환경 셋팅
```
conda create -n LVS python=3.8  
conda activate LVS  
pip install opencv-python  
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts  
pip install segmentation-models-pytorch albumentations  
```

## 데이터셋 준비
다음과 같이 data 폴더 구성
```
LeftVentricleSegmentation
├── data
│   └── echocardiography
│       ├── train
│       │   ├── A2C
│       │   └── A4C
│       ├── validation
│       │   ├── A2C
│       │   └── A4C
│       └── test
│           ├── A2C
│           └── A4C
├── dataset
├── experiments
├── scripts
├── utils
.
.
.
```


# 모델 학습
모델 학습 예시
```
python train.py --dataset A2C\
                --encoder resnet18\
                --num-workers 4\
                --batch-size 4\
                --lr 0.001\
                --wd 0.01\
                --max-epochs 60
```
- dataset : A2C 또는 A4C  
- encoder : resnet18 또는 se_resnext50_32x4d  

모델의 체크포인트는 다음과 같은 경로에 저장됨  
```
./results/checkpoints/[dataset]_[encoder]_lr_[lr]_wd_[wd]_epochs_[max-epochs].pth  
ex) ./results/checkpoints/A2C_resnet18_lr_0.001_wd_0.01_epochs_60.pth
```



# 모델 테스트
하나의 모델 테스트
```
python test.py [checkpoint path] --mode validation
```
예시  
```
python test.py ./results/checkpoints/A2C_resnet18_lr_0.001_wd_0.01_epochs_60.pth --mode validation
```
앙상블 모델 테스트
```
python test_ensemble.py [checkpoint1 checkpoint2 ...] --mode validation
```
예시  
```
python test_ensemble.py ./results/checkpoints/A2C_resnet18_lr_0.001_wd_0.01_epochs_60.pth ./results/checkpoints/A2C_se_resnext50_32x4d_lr_0.0001_wd_0.001_epochs_60.pth --mode validation
```


# 최종 제출 모델 테스트
## validation 데이터셋
- A2C
    ```
    ./scripts/test_A2C_validation.sh
    ```
- A4c
    ```
    ./scripts/test_A4C_validation.sh
    ```

    결과  
    ||DSC|JI|
    |---|---|---|
    |A2C|0.9567|0.9170|
    |A4C|0.9740|0.9510|

## test 데이터셋
- A2C
    ```
    ./scripts/test_A2C_test.sh
    ```
- A4c
    ```
    ./scripts/test_A4C_test.sh
    ```
