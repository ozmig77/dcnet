# Dual Composition Network (DCNet)

This project hosts the code for our paper.

- [Jongseok Kim](https://ozmig77.github.io/), [Youngjae Yu](https://yj-yu.github.io/home), Hoeseong Kim and [Gunhee Kim](http://vision.snu.ac.kr/gunhee/).
Dual Compositional Learning in Interactive Image Retrieval. In *AAAI*, 2021.

This project is an Winning Solution in [FashionIQ 2020](https://sites.google.com/view/cvcreative2020/fashion-iq).

## Reference

If you use this code as part of any published research, please refer following paper,

```bibtex
@inproceedings{kim:2021:AAAI,
    title="{Dual Compositional Learning in Interactive Image Retrieval}",
    author={Kim, Jongseok and Yu, Youngjae and Kim, Hoeseong and Kim, Gunhee},
    booktitle={AAAI},
    year=2021
}
```

## Getting Started

### Prerequisites
Language: python\==3.7.7, pytorch\==1.4.0
```
pip install -r requirement.txt
```

### Datasets
- Download Fashion-IQ dataset images from [here](https://github.com/hongwang600/fashion-iq-metadata). Save it under ./dataset/fashioniq/images/   
(May use `preprocess/download_fashioniq.py` for downloading images.)

- Download Fashion-IQ dataset annotations from [here](https://github.com/XiaoxiaoGuo/fashion-iq). Save it under ./dataset/fashioniq/
(Files under folder `captions` and `image_splits`)                                                                                                

- Download Shoe dataset images from [here](http://tamaraberg.com/attributesDataset/attributedata.tar.gz). Save it under ./dataset/shoe/images/

- Download Shoe dataset annotations from [here](https://github.com/yanbeic/VAL/tree/master/datasets/shoes). Save it under ./dataset/shoe/         
(We only need `relative_captions_shoes.json`, `eval_im_names.txt`, and `train_im_names.txt`.)


### Preprocessing
Below code resize images and create glove embedded caption files.
```
cd preprocess
python -m nltk.downloader 'punkt'
python -m spacy download en_vectors_web_lg
python process_cap.py
python gen_shoe_cap.py
python resize_img.py
```


## How to Run the code

### Evaluation
For evaluation, first download checkpoint and config.json file from [Fashion IQ](https://drive.google.com/drive/folders/1wgygqF095Di67EaHaGOXbwh3wEzk9izB?usp=sharing), [Shoe](https://drive.google.com/drive/folders/1saN1IhZ_fGOTfRMhoJ6QVVvi8Vj57mZn?usp=sharing) under `./logdir/fashioniq_dcnet`, `./logdir/shoe_dcnet` respectively.
Then run below,
```
python test.py --resume logdir/fashioniq_dcnet/
python test.py --resume logdir/shoe_dcnet/
```
### Training
For training run below,
```
python train.py --config configs/ce/fashioniq_dcnet.json --logdir fashioniq
python train.py --config configs/ce/shoe_dcnet.json --logdir shoe
```
You can change json file for other settings.

### Experiments
For qualitative result and attention plot for text expert, refer
```
./experiments/result.ipynb
./experiments/textatt.ipynb
```

## Deepfashion pretrained densenet
For Fashion IQ challenge performance, we provide deepfashion pretrained densenet as backbone (you may need larger VRAM)
1. Download backbone checkpoint from [here](https://drive.google.com/file/d/1L5ArT7n-D4bB9QkmntJQNWfPYgu0UFCX/view?usp=sharing) and save it under ./deepfashion/logdir/deepfashion_densenet
2. Run following command to train
```
python train.py --config configs/ce/fashioniq_dcnet_deep.json --logdir fashioniq_deep
```
