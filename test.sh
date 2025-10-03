#$ -l tmem=32G
#$ -l h_vmem=32G
#$ -l h_rt=0:10:00
#$ -l gpu=true

#$ -S /bin/bash
#$ -j y
#$ -N Test
#$ -V
#$ -wd /home/mmasramo/ProsRegNet

hostname

date

export PATH=/share/apps/python-3.6.9-tkinter/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.6.9-tkinter/lib:$LD_LIBRARY_PATH

python3 register_images.py -i jsonData/TCIA_FUSION.json     -v -pm -pf -r -n hist-mri-dwi2

date

