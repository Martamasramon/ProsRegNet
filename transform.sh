#$ -l tmem=32G,h_vmem=32G
#$ -l h_rt=2:00:00
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

python3 transorm_high_res.py 

date
