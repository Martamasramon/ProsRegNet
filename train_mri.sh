#$ -l tmem=16G,h_vmem=16G
#$ -l h_rt=0:20:00
#$ -l gpu=true

#$ -S /bin/bash
#$ -j y
#$ -N TrainMRI
#$ -V
#$ -wd /home/mmasramo/ProsRegNet

hostname

date

export PATH=/share/apps/python-3.6.9-tkinter/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.6.9-tkinter/lib:$LD_LIBRARY_PATH

python3 train.py --geometric-model tps-mri -n '3x3' -t 'mri_train.csv' -s 'mri_test.csv'
python3 train.py --geometric-model affine  -n 'mri'  
python3 register_mri.py -i jsonData/TCIA_FUSION_T2-DWI.json -v -pm -pf -r  -n '3x3'

# python3 train.py --geometric-model tps      -n 'mri_high_res' -t 'mri_train.csv' -s 'mri_test.csv'
# python3 train.py --geometric-model affine   -n 'mri_high_res' 

date
