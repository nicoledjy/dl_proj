# dl_proj

- AE pretraining



- Roadmap prediction:

  Use folder `hrnet`

  - --batch-size, default=2
  - --epochs, default=10
  - --lr, default=1e-4
  - --weight-decay, default=1e-4
  - --data-dir, default='../data'
  - --out-file, default='HRNET_RM_model.pt'

  ```
  python train_HRNet_RoadMap.py
  ```



- Bounding Box prediction:

  Use folder `yolo`

  ```
  python trainYolo_withPretrain.py 
  python trainYolo.py 
  ```



- Object detection without per-training:

  Use folder `without_pretrain`, arguments for the `main.py` are the following:

  - --batch-size, default=2
  - --epochs, default=10
  - --lr, default=1e-4
  - --weight-decay, default=1e-4

  ```
  python main.py
  ```





- PIRL pre-training:

  Use folder  `pretrain`, run `pirl_train.py`(path change may be required) , and main arguments are the following:

  - --num-scene, number of scenes used for the pertaining, default=106
  - --model-type, default=res18
  - --batch-size, default=128
  - --epochs, default=100
  - --lr, default=0.01
  - --count-negatives, default=6400 **(need to be half the size of images used)**

  ```
  python pirl_train.py --num-scene 1 --model-type res50 --batch-size 2 --lr 0.1 --count-negatives 200
  ```





- Object detection with Resnet50 backbone architecture:

  - use folder `pretrain_obj`
  - No argument parser added yet, need to set pretrain_res=True and add the path in `main.py`

  ```
  python main.py
  ```
