# for densenet
# CUDA_VISIBLE_DEVICES=0 python main.py --epoch 300 --batch-size 64 -ct 100
# python main.py --epoch 300 --batch-size 64 -ct 100
# python main.py --epoch 300 --batch-size 48 -ct 100

# for resnet
#python main.py --epoch 160 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 1e-4 -ct 10
python main.py --epoch 100 --batch-size 512 -ct 100
