# 模型压缩全家桶之知识蒸馏
## <div align="center"> 介绍</div>

yolov5-l模型压缩至yolov5-s
压缩算法是 https://github.com/twangnh/Distilling-Object-Detectors 

## <div align="center">Quick Start Examples</div>


<details open>
<summary>Install</summary>

Python >= 3.6.0 required with all [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) dependencies installed:
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->
```bash
$ git clone https://github.com/magicshuang/yolov5_distillation
$ cd magicshuang/yolov5_distillation
$ pip install -r requirements.txt
```
</details>









<details>
  
<summary>Training</summary>

没有teacher模型的人，先训练teacher模型
```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                40
                                         yolov5l                                24
                                         yolov5x                                16
```
</details>  




<details open>
<summary>Distillation Training</summary>
  
```bash
$ python Distill_train.py --data coco.yaml --teacher-weights [your path] --batch-size 64
  
```

</details>




## <div align="center">详细了解</div>

比yolov5源码只多了两个文件
Distill_train.py & utils/distill_fun.py

所以可以直接下载这两个文件就可以了





