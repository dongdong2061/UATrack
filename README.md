# UATrack

Pytorch implementation of the paper "Uncertainty-aware RGBT Tracking".

## Introduction
![Arch](img\overallnetwork.png)
- UATrack leverages uncertainty to enhance robust representation learning and reliable multimodal fusion for RGBT tracking, effectively addressing the challenges arising from dynamically varying modality quality in complex scenarios.
- UATrack achieves SOTA result on GTOT, RGBT210, RGBT234 and LasHeR datasets.

Note: This is an extended version of our conference paper [TUMFNet](https://github.com/dongdong2061/IJCAI25-TUMFNet) published at IJCAI 2025.

### Models and Results
You can download the model and results from [here](https://pan.baidu.com/s/1Pl4VFVA8VVHt0Eu132ucrA?pwd=RGBT) [Extraction Code: RGBT].



### Path Setting
Run the following command to set paths:
```
cd <PATH>
python tracking/create_default_local_file.py --workspace_dir . --data_dir <PATH_of_Datasets> --save_dir ./output
```
You can also modify paths by these two files:
```
./lib/train/admin/local.py  # paths for training
./lib/test/evaluation/local.py  # paths for testing
```

### Training
Dowmload the pretrained [foundation model](https://www.kaggle.com/datasets/zhaodongding/drgbt603-results/data) (OSTrack and DropMae)
and put it under ./pretrained/.
```
CUDA_VISIBLE_DEVICES=0,1  NCCL_P2P_LEVEL=NVL nohup  python tracking/train.py --script drgbt --config DRGBT603 --save_dir ./output --mode multiple --nproc_per_node 1 >  train_track.log &
```

### Test
```
bash eval_rgbt.sh
```

### Evaluation for LasHeR, RGBT234, RGBT210 and GTOT
```
python eval_lasher.py
python eval_rgbt234.py
python eval_rgbt210.py
python eval_gtot.py
```

## Acknowledgment
- This repo is based on [BAT](https://github.com/SparkTempest/BAT) which is an exellent work, helps us to quickly implement our ideas.
- Thanks for the [OSTrack](https://github.com/botaoye/OSTrack) and [PyTracking](https://github.com/visionml/pytracking) library.
