#$ -l tmem=32G,h_vmem=32G
#$ -l h_rt=0:30:00
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

python3 train.py --geometric-model tps -n 'fIC' -t 'fIC_TPS_train.csv' -s 'fIC_TPS_test.csv'


date
