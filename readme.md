## 函式庫安裝

```
torch請上官網看
cd requirement
pip install opencv_python-4.7.0.72-cp37-abi3-win_amd64.whl #py3.9
pip install opencv_python-4.5.4.60-cp310-cp310-win_amd64.whl #py3.10
pip install tensorrt_dispatch-10.0.0b6-cp39-none-win_amd64.whl #可以不用
pip install tensorrt_lean-10.0.0b6-cp39-none-win_amd64.whl #可以不用
pip install tensorrt-10.0.0b6-cp39-none-win_amd64.whl #可以不用
conda install huggingface::transformers --y
conda install conda-forge::sklearn-contrib-lightning --y
conda install conda-forge::matplotlib --y
conda install conda-forge::dill --y
conda install conda-forge::pillow --y
conda install conda-forge::ultralytics --y
conda install conda-forge::opencv --y
conda install conda-forge::supervisor --y
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
conda install conda-forge::onnxruntime --y
conda install conda-forge::onnx --y
conda install conda-forge::ipython --y
conda install anaconda::pillow --y
pip install dxcam #截圖用 
conda install conda-forge::pyautogui --y
conda install conda-forge::pytube --y
```

# keypoints”: [“nose”,“left_eye”,“right_eye”,“left_ear”,“right_ear”,

# “left_shoulder”,“right_shoulder”,“left_elbow”,“right_elbow”,“left_wrist”,

# “right_wrist”,“left_hip”,“right_hip”,“left_knee”,“right_knee”,“left_ankle”,“right_ankle”]

# names: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',

# 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',

# 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',

# 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',

# 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',

# 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',

# 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',

# 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

請參考ppt