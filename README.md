# SemiCOL-Challenge
SemiCOL Challenge 2023 - Semi-supervised learning for colorectal cancer detection

# Create environment:

### Create conda env:
``` shell
conda create --name semicol python=3.8 pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
source activate semicol
```

### Install requirements:
``` shell
pip install -r requirements.txt
```

# Run instructions:


### Patch and create .h5 from training dataset:
``` shell
python3 create_dataset.py \
        --manual_path="/home/data/semicol/DATASET_TRAIN/01_MANUAL/" \
        --output="/home/data/semicol/DATASET_TRAIN/semicol.h5"
```

### Stain normalisation:
``` shell
python3 create_h5_stain_norm.py \
        --h5-location="/home/data/semicol/DATASET_TRAIN/semicol.h5" \
        --normalisation-type="Macenko" \
        --target-image="/home/cggm1/data/semicol/DATASET_TRAIN/01_MANUAL/DS_M_1/ukk_case_04/image/ukk_case_04 [d=2.16945,x=91117,y=78100,w=6508,h=6509].png"
```

### Train model:
``` shell
CUDA_VISIBLE_DEVICES=0 python3 train.py \
        --data_path="/home/cggm1/data/semicol/DATASET_TRAIN_mini/semicol_mini.h5" \
        --architecture=unet \
        --batch_size=128 \
        --classes=10 \
        --epochs=5 \
        --index_to_ignore=0 \
        --learning_rate=0.0001 \
        --normalise=False --optim_type=adam \
        --patch_dim=256 \
        --weights=rebalanced \
        --amp \
        --class_reduce=True
```

### Model inference (segmentation):
``` shell
CUDA_VISIBLE_DEVICES=0 python3 validation_inference_part_1.py \
        --input="/home/cggm1/data/semicol_docker/input/01_MANUAL" \
        --output="/home/cggm1/data/semicol/DATASET_VAL/docker/" \
        --model_path="/home/cggm1/data/semicol/checkpoints/unet/rebalanced/256/drb8at9y/checkpoint_epoch34.pth" \
        --architecture="unet"
        
#working example below

CUDA_VISIBLE_DEVICES=0 python3 validation_inference_part_1.py --input="/home/cggm1/data/semicol_docker/input/01_MANUAL" --output="/home/cggm1/data/semicol_docker/pred" --model_path="/home/cggm1/data/semicol/checkpoints/unet/rebalanced/256/mjdx91bl/checkpoint_epoch5.pth" --architecture="unet"
```

### Model inference (Tumour classification):
**First, generate a list of patches:**
``` shell
python3 create_patches.py --source="/home/cggm1/data/semicol_docker/input/02_BX/" --save_dir="/dir/to/save/h5/sets/" --preset="semicol.csv" --patch
```
Note: If the above does not run with the current environment, create a new temprary conda env with python=3.8 using ``conda create -n patching python=3.8`` then run ``python -m pip install -U tifffile[all]`` inside that environment.

**Then, use the list of generated patches to get a slide level classification**

``` shell
CUDA_VISIBLE_DEVICES=0 python3 validation_inference_part_2_01.py \
        --coords="./patches/" \
        --input_data="/home/cggm1/data/semicol_docker/input/02_BX" \
        --model_path="/home/cggm1/data/semicol/checkpoints/unet/rebalanced/256/drb8at9y/checkpoint_epoch34.pth" \
        --num_classes=10
```

**Finally, convert class count metrics to slide level classification:**
``` shell
python3 validation_inference_part_2_02.py
```
You may wish to make adjustments to the thresholds in this file in order to optimise classification performance.