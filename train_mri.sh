#$ -l tmem=16G,h_vmem=16G
#$ -l h_rt=0:10:00
#$ -l gpu=true

#$ -S /bin/bash
#$ -j y
#$ -N Train MRI
#$ -V
#$ -wd /home/mmasramo/ProsRegNet

hostname

date

export PATH=/share/apps/python-3.6.9-tkinter/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.6.9-tkinter/lib:$LD_LIBRARY_PATH

python3 train.py --geometric-model tps -n 'mri' -t 'mri_train.csv' -s 'mri_test.csv' -p 'datasets/training'

date
