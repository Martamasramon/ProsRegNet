#$ -l tmem=32G,h_vmem=32G
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

python3 register_images.py -i jsonData/TCIA_FUSION.json     -v -pm -pf -r  -n 'hist-mri-dwi2'
python3 register_images.py -i jsonData/TCIA_FUSION_fIC.json -v -pm -pf -r  -n fIC
python3 register_t2_dwi.py -i jsonData/TCIA_FUSION_mri.json -v -pm -pf -r  -n '3x3'
# -v: verbose, -pf: process fixed (MRI), -pm: process moving (histo), -r: register

python3 transform_histo.py -m 'T2'
python3 transform_histo.py -m 'fIC'
python3 transform_histo.py -m 'b90'

date

