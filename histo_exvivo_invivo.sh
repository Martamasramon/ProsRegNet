#$ -l tmem=32G,h_vmem=32G
#$ -l h_rt=0:10:00
#$ -l gpu=true

#$ -S /bin/bash
#$ -j y
#$ -N Hist_exvivo_invivo
#$ -V
#$ -wd /home/mmasramo/ProsRegNet

hostname

date

export PATH=/share/apps/python-3.6.9-tkinter/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.6.9-tkinter/lib:$LD_LIBRARY_PATH

python3 register_images.py -i jsonData/TCIA_FUSION.json -v -pm -pf -r  -n 'hist-mri-dwi2'
python3 register_exvivo_invivo.py -i jsonData/TCIA_FUSION.json -v -pm -r  -n 'pretrained'

python3 transform_histo_exvivo_invivo.py -m 'T2'

date

