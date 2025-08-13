from ultralytics import YOLO

def main()-> None:
    model_path = r'runs\detect\train\weights\best.onnx'
    class_yaml = 'data\data.yaml'
    image_path = r'bb6ababa53f94b9922bd2e68b7e07f40.jpg'
    model_yolov11 = YOLO(model_path)
    model_yolov11.predict(image_path,device='cpu')

if __name__ =="__main__":
    main()
