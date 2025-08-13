from ultralytics import YOLO

def test_simulate(model_path: str, data_path: str):
    best_model = YOLO(model_path)
    metrics = best_model.val(
        data=data_path,
        split='test',
        batch=4,
        imgsz=640,
        device='cuda'
    )
    print(metrics)
    best_model.predict("bb6ababa53f94b9922bd2e68b7e07f40.jpg")

def saving_model(model_path: str, export_path: str):
    loaded_model = YOLO(model_path)
    loaded_model.export(format='onnx',device='cpu')

if __name__ == "__main__":
    best_model_path = r"runs\detect\train2\weights\best.pt"
    data_path = r"data\data.yaml"
    export_path = r"runs\detect\train2\weights\best.onnx"

    test_simulate(model_path=best_model_path, data_path=data_path)
    saving_model(best_model_path, export_path)
    test_simulate(model_path=best_model_path, data_path=data_path)
