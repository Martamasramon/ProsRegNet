#$ -l tmem=32G,h_vmem=32G
#$ -l h_rt=0:10:00
#$ -l gpu=true

#$ -S /bin/bash
#$ -j y
#$ -N TestMRI
#$ -V
#$ -wd /home/mmasramo/ProsRegNet

hostname

date

export PATH=/share/apps/python-3.6.9-tkinter/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.6.9-tkinter/lib:$LD_LIBRARY_PATH

python3 register_mri.py -i jsonData/TCIA_FUSION_T2-DWI.json -v -pm -pf -r  -n 'mri-3x3'
# -v: verbose, -pf: process fixed (MRI), -pm: process moving (histo), -r: register

date

