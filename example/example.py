from ultralytics import YOLO

def main()-> None:
    model_path = r'runs\detect\train\weights\best.onnx'
    class_yaml = 'data\data.yaml'
    image_path = r'e53d1b71b38ed9b528649b7541c8bba7.jpg'
    model_yolov11 = YOLO(model_path)
    model_yolov11.predict(image_path,device='cpu')

if __name__ =="__main__":
    main()
