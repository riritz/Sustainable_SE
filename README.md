# Sustainable_SE

### How to run the experiment:

1. Download dataset  
   - Create a folder named `dataset` inside the `experiment` folder
   - Download the COCO dataset at http://images.cocodataset.org/zips/val2017.zip
   - Unzip to `dataset` folder

2. Download model weights
- YOLOv8 - medium (25.9)
```
cd experiment 
#Download weights 
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m.pt
```
- RFDETR - nano (30.5 param)
```
cd experiment 
#Download weights 
wget wget https://huggingface.co/roboflow/rf-detr/resolve/main/rf-detr-nano.pth
```

3. Install requirements:
```
pip install -r requirements.txt
```

4. Run the script:
```
cd experiment

# Make script executable
chmod +x run_experiment.sh

# Run script
./run_experiment.sh
```
