## MLOps Computer Vision project
Simple computer vision project deployed using FastAPI and Docker.

The app offers a detection and a segmentation endpoint.

The Deep learning models used for this projects are [YoloV8](https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modeshttps://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes) models from Ultralytics.

### Docker usage
- build image: ```docker build -t container-name .```
- run container: ```docker run -p 8000:port container-name```


### Custom models
You can deploy your own YoloV8 custom models simply modifying the .pt files in the model.py script and adding them in the models folder.