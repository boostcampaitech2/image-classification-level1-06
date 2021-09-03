<div align="center">
  <h1>pstage_01_image_classification</h1>
</div>

![pstage](https://user-images.githubusercontent.com/51067412/132005999-5cb1ecf0-7992-4ee6-8788-567244d00dbf.PNG)

## :fire: Getting Started
### Dependencies
- torch==1.6.0
- torchvision==0.7.0                                                              

### Install Requirements
- `pip install -r requirements.txt`

### Contents
- `dataset.py`
- `face_image.py` : FaceNet 적용
- `loss.py` 
- `model.py` 
- `optimizer.py` 
- `train.py`
- `inference.py`
- `evaluation.py`
- `train_stratified_face.csv` : stratified dataset (train)
- `valid_stratified_face.csv` : stratified dataset (valid)
- `perfect_train.csv` : dataset for multilabel classification

### Training
- `SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python train.py`

### Inference
- `SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] python inference.py`

### Evaluation
- `SM_GROUND_TRUTH_DIR=[GT dir] SM_OUTPUT_DATA_DIR=[inference output dir] python evaluation.py`

## :mag: Overview
### Background
> COVID-19의 확산으로 우리나라는 물론 전 세계 사람들은 경제적, 생산적인 활동에 많은 제약을 받고있습니다. </br>
> 확산을 막기위해 많은 노력들을 하고 있지만 COVID-19의 강한 전염력 때문에 우리를 오랫동안 괴롭히고 있습니다. </br>
> 이를 해결하는 방법은 모든 사람이 마스크로 코와 입을 가려서 혹시 모를 감염자로부터의 전파 경로를 원천 차단하는 것입니다. </br>
> 이를 위해 우리는 공공장소에서 모든 사람들의 올바른 마스크 착용 상태를 검사하는 시스템이 필요합니다. </br>
> 즉, **카메라로 비춰진 사람 얼굴 이미지만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, </br>
> 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템이 필요합니다.**

### Problem definition
> 카메라로 비춰진 사람 얼굴 이미지만으로 이 사람이 마스크를 쓰고 있는지, </br>
> 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템 or 모델

### Development environment
- GPU V100 원격 서버
- PyCharm 또는 Visual Studio Code | Python 3.7(or over)

### Evaluation
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F6390139%2Fb19f3db709b41788c3b1333ef1ae11a9%2Ff1score.png?generation=1608093256720406&alt=media">

## Dataset Preparation
### Prepare Images
- 전체 사람 수 : 4500명 (train : 2700 | eval : 1800)
- age : 20대 - 70대
- gender : 남,여
- mask : 개인별 정상 착용 5장, 비정상적 착용 1장(코스크,턱스크...), 미착용 1장
- 전체 31,500 Images (train : 18,900 | eval : 12,600)
- 이미지 크기 : (384,512)

### Data Labeling
- mask, gender, age 기준 18개의 클래스로 분류
<img src="https://user-images.githubusercontent.com/68593821/131881060-c6d16a84-1138-4a28-b273-418ea487548d.png" height="500"/>

### [Facenet](https://arxiv.org/pdf/1503.03832.pdf)
 - [face_image.py](https://github.com/boostcampaitech2/image-classification-level1-06/blob/main/face_image.py)

```
$ python face_image.py
```
  - FaceNet 이용 얼굴 인식 후 FaceCrop
  - FaceCrop 후 Resize를 거친 이미지 크기 : (280, 210)

### Generate CSV files
- `perfect_train.csv`

## :running: Training
`SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python train.py`

### Train Models
- [EfficientNet](https://arxiv.org/pdf/1905.11946.pdf)
  - EfficientNet_b2_pruned 
  - EfficientNet_b4
- [SEResNet(SE block + ResNet)](https://arxiv.org/pdf/1709.01507.pdf)
  - SEResNet26d_32x4d
- [NFNet](https://arxiv.org/pdf/2102.06171.pdf)
  - NFNet_l0

### Stratified K-fold
```py 
from sklearn.model_selection import StratifiedKFold
```
`python train.py --cv True`<br>
`train.py` cross_validation 함수

### Multilabel Classification
- gender, age, mask 각각 학습
- `train.py` multi_train 함수


## :thought_balloon: Inference
`SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] python inference.py`

```
# 단일 model을 통해 inference 시
$ python inference.py \
  --model_name={kinds of models} \
  --model_dir={model_filepath} \
  --pth_name={model parameter name (ensemble, cross_validation : best)} \
  --output_name={output_filename} \
```

```
# cross_validation 을 사용해 나온 model 5개를 통해 inference 시
$ python inference.py \
  --model_name={kinds of models} \
  --model_dir={model_filepath} \
  --pth_name=best \
  --output_name={output_filename} \
  --cv=True
```

```
# ensemble 을 사용해 inference 시
$ python inference.py \
  --ensemble_model_name={kinds of models : 띄어쓰기로 구분한 여러개의 model_name} \
  --ensemble_model_dir={model_filepath : 띄어쓰기로 구분한 여러개의 model_file_path} \
  --pth_name=best \
  --output_name={output_filename} \
  --ensemble=True
```

```
# multilabel inference 시
$ python inference.py \
  --model_name={kinds of models} \
  --model_dir={model_filepath} \
  --pth_name=best \
  --output_name={output_filename} \
  -- multi True
```
