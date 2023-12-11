# DiffMOT
## DiffMOT: A Real-time Diffusion-based Multiple Object Tracker with Non-linear Prediction 
![Teaser](assets/teaser_git.png)

## Framework
![Framework](assets/diffmot_git.png)
![Framework](assets/ddmp_git.png)

## I. Installation.
1. install torch
~~~
conda create -n diffmot python=3.9
conda activate diffmot
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
~~~
2. install other packages.
~~~
pip install -r requirement.txt
~~~

## II. Prepare Data.
The file structure should look like:  
(a) DanceTrack:
~~~
{DanceTrack ROOT}
|-- dancetrack
|   |-- train
|   |   |-- dancetrack0001
|   |   |   |-- img1
|   |   |   |   |-- 00000001.jpg
|   |   |   |   |-- ...
|   |   |   |-- gt
|   |   |   |   |-- gt.txt            
|   |   |   |-- seqinfo.ini
|   |   |-- ...
|   |-- val
|   |   |-- ...
|   |-- test
|   |   |-- ...
~~~
(b) SportsMOT:
~~~
{SportsMOT ROOT}
|-- sportsmot
|   |-- splits_txt
|   |-- scripts
|   |-- dataset
|   |   |-- train
|   |   |   |-- v_1LwtoLPw2TU_c006
|   |   |   |   |-- img1
|   |   |   |   |   |-- 000001.jpg
|   |   |   |   |   |-- ...
|   |   |   |   |-- gt
|   |   |   |   |   |-- gt.txt
|   |   |   |   |-- seqinfo.ini         
|   |   |   |-- ...
|   |   |-- val
|   |   |   |-- ...
|   |   |-- test
|   |   |   |-- ...
~~~
(c) MOT17/20:
~~~
{MOT17/20 ROOT}
|-- mot
|   |-- train
|   |   |-- MOT17-02
|   |   |   |-- img1
|   |   |   |   |-- 000001.jpg
|   |   |   |   |-- ...
|   |   |   |-- gt
|   |   |   |   |-- gt.txt            
|   |   |   |-- seqinfo.ini
|   |   |-- ...
|   |   |-- MOT20-01
|   |   |   |-- img1
|   |   |   |   |-- 000001.jpg
|   |   |   |   |-- ...
|   |   |   |-- gt
|   |   |   |   |-- gt.txt            
|   |   |   |-- seqinfo.ini
|   |   |-- ...
|   |-- test
|   |   |-- ...
~~~

and run:

```
python dancetrack_data_process.py
python sports_data_process.py
python mot_data_process.py
```

## III. Training.
* Change the data_dir in config
* Train on DanceTrack, SportsMOT, and MOT17/20:
```
python main.py --config ./configs/dancetrack.yaml
python main.py --config ./configs/sportsmot.yaml
python main.py --config ./configs/mot.yaml
```

## IV. Tracking.
* Change the det_dir, info_dir, reid_dir, and save_dir in config
* Track on DanceTrack, SportsMOT, MOT17, and MOT20:
```
python main.py --config ./configs/dancetrack_test.yaml
python main.py --config ./configs/sportsmot_test.yaml
python main.py --config ./configs/mot17_test.yaml
python main.py --config ./configs/mot20_test.yaml
```

## Concat
If you have some questions, please concat with kroery@shu.edu.cn.

## Acknowledgement
A large part of the code is borrowed from [DDM-Public](https://github.com/GuHuangAI/DDM-Public) and [Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT). Thanks for their wonderful works.

## Citation
~~~
waiting for updating
~~~



