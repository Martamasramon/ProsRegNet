#$ -l tmem=32G,h_vmem=32G
#$ -l h_rt=0:10:00
#$ -l gpu=true

#$ -S /bin/bash
#$ -j y
#$ -N Train
#$ -V
#$ -wd /home/mmasramo/ProsRegNet

hostname

date

export PATH=/share/apps/python-3.6.9-tkinter/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.6.9-tkinter/lib:$LD_LIBRARY_PATH

#python3 train.py --geometric-model affine -n 
python3 train.py --geometric-model tps -n 'default'         -t 'train.csv'              -p 'datasets/training'
#python3 train.py --geometric-model tps -n '47'              -t 'train_updated_47.csv'   -p 'datasets/training'
#python3 train.py --geometric-model tps -n 'downsampled_47'  -t 'train_updated_47.csv'   -p 'datasets/training_downsampled'

date
