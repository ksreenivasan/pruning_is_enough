# set up ffcv script

docker run --gpus all --ipc=host --name kartik_imagenet -it -v /hdd1/ILSVRC2012:/data/imagenet -v /hdd3/ksreenivasan:/workspace nvcr.io/nvidia/pytorch:21.12-py3

sudo docker pull nvcr.io/nvidia/pytorch:22.01-py3

# to fix dependencies for ffcv
apt-get install ffmpeg libsm6 libxext6  -y

# clone repo
git clone git@github.com:libffcv/ffcv-imagenet.git

# install dependencies
cd ffcv-imagenet
pip install -r requirements.txt

# set environment variables
export IMAGENET_DIR=/data/imagenet
export WRITE_DIR=/workspace/ffcv-imagenet/data
mkdir data

# create ffcv data

# Serialize images with:
# - 500px side length maximum
# - 50% JPEG encoded, 90% raw pixel values
# - quality=90 JPEGs
./write_imagenet.sh 500 0.50 90


python train_imagenet.py --config-file rn18_configs/rn18_88_epochs.yaml \
    --data.train_dataset=/workspace/ffcv-imagenet/data/train_500_0.50_90.ffcv \
    --data.val_dataset=/workspace/ffcv-imagenet/data/val_500_0.50_90.ffcv \
    --data.num_workers=12 --data.in_memory=1 \
    --logging.folder=/workspace/ffcv-imagenet/logs

# start exec: 5:24pm (with batch size 512)