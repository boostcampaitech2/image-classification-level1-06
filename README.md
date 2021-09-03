# pstage_01_image_classification

## Getting Started    
### Dependencies
- torch==1.6.0
- torchvision==0.7.0                                                              

### Install Requirements
- `pip install -r requirements.txt`

### Training
- `SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python train.py`

### Inference
- `SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] python inference.py`

### Evaluation
- `SM_GROUND_TRUTH_DIR=[GT dir] SM_OUTPUT_DATA_DIR=[inference output dir] python evaluation.py`

### Contents
- `dataset.py`
- `face_image.py`
- `loss.py`
- `model.py`
- `optimizer.py`
- `train.py`
- `inference.py`
- `evaluation.py`
- `train_stratified_face.csv`
- `valid_stratified_face.csv`

## Dataset Preparation
### Prepare Images
- 전체 사람 수 : 4500명 (train : 2700 | eval : 1800)
- age : 20대 - 70대
- gender : 남,여
- mask : 개인별 정상 착용 5장, 비정상적 착용 1장(코스크,턱스크...), 미착용 1장
- 전체 31,500 Images (train : 18,900 | eval : 12,600)
- 이미지 크기 : (384,512)

### Data Labeling
<img src="https://user-images.githubusercontent.com/68593821/131881060-c6d16a84-1138-4a28-b273-418ea487548d.png" height="500"/>

### Facenet
 - [face_image.py](https://github.com/boostcampaitech2/image-classification-level1-06/blob/main/face_image.py)

```
$ python face_image.py
```

### Generate CSV files
- 'perfect_train.csv'

## Training
`SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python train.py`

### Train Models
- efficientnet_b2_pruned 
- efficientnet_b4
- seresnext26d_32x4d
- nfnet_l0

### Stratified K-fold
### Multilabel Classification
- gender
- age
- mask


## Inference
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

