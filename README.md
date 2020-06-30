# Carla-Visual-Relocalization

Generate dataset for visual relocalization in CARLA 0.8.2

## 1 Generate Dataset in Carla

* Find the Two CARLA scripts, then add it to **CARLA_0.8.2/PythonClient/scripts/CARLA**

  [RelocalizationQuery.py](https://github.com/FiftyWu/Carla-Visual-Localization/blob/master/CARLA/RelocalizationQuery.py)

  [RelocalizationReference.py](https://github.com/FiftyWu/Carla-Visual-Localization/blob/master/CARLA/RelocalizationReference.py)

* Start carla simulator in Town01 or Town 02 (default Town01) with **PowerShell**

  ```powershell
  .\CarlaUE4.exe -carla-server -windowed -ResX=400 -ResY=300  -benchmark -fps=10
  .\CarlaUE4.exe  /Game/Maps/Town02  -carla-server -windowed -ResX=400 -ResY=300  -benchmark -fps=10
  ```

* Set up **weather, playerstart, vehicles, pedestrians** etc. then run [RelocalizationReference.py](https://github.com/FiftyWu/Carla-Visual-Localization/blob/master/CARLA/RelocalizationReference.py)
* Set up **weather, playerstart, vehicles, pedestrians** etc. then run [RelocalizationQuery.py](https://github.com/FiftyWu/Carla-Visual-Localization/blob/master/CARLA/RelocalizationQuery.py)

* **Output format**

| folder name              | description                                 |
| ------------------------ | ------------------------------------------- |
| e.g. W000_P100_V000_P000 | weather, playerstart, vehicles, pedestrians |
| W000_P100_V050_P200      | as above                                    |
| W000_P100_V075_P300      | as above                                    |

| subfolder name        | description        |
| --------------------- | ------------------ |
| /Depth                | key frame,256*256  |
| /RGB                  | key frame,256*256  |
| /SemanticSegmentation | key frame,256*256  |
| Control.txt           | each frame,256*256 |
| Trajectory.txt        | each frame,256*256 |

## 2 Use Dynamic2static Model to infer

* Run [CarlaProcessed.ipynb](https://github.com/FiftyWu/Carla-Visual-Relocalization/blob/master/CarlaProcessed.ipynb)

* **Output format**

| folder name | description                                        |
| ----------- | -------------------------------------------------- |
| /AB         | dynamic,static, width=256*2, height=256            |
| /ABC        | dynamic,static,segmantic mask, width=256*3, height |

* Run **test_with_mask** to infer, then download to local pc from remote server

> pretrained Dynamic2static  model

* **Output format**

| folder name | description             |
| ----------- | ----------------------- |
| /images     | fake_B, Gx, real_A,.... |

* Run  [images2processed.ipynb](https://github.com/FiftyWu/Carla-Visual-Relocalization/blob/master/images2processed.ipynb)

> deltete Gx, real_A,..., rename fake_B.png to xxxxxx.png

## 3 Evaluate Precision of Visual ReLocalization

* In pycharm run [ImageRetrieval.py](https://github.com/FiftyWu/Carla-Visual-Relocalization/blob/master/ImageRetrieval.py)

* in jupyter notebook run  [get_precision.ipynb](https://github.com/FiftyWu/Carla-Visual-Relocalization/blob/master/get_precision.ipynb) (recommended)
