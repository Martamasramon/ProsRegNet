#$ -l tmem=32G,h_vmem=32G
#$ -l h_rt=0:00:30
#$ -l gpu=true

#$ -S /bin/bash
#$ -j y
#$ -N Transform
#$ -V
#$ -wd /home/mmasramo/ProsRegNet

hostname

date

export PATH=/share/apps/python-3.6.9-tkinter/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.6.9-tkinter/lib:$LD_LIBRARY_PATH

python3 transform_histo.py -m 'T2'
python3 transform_histo.py -m 'b90'
python3 transform_histo.py -m 'b0'

date
