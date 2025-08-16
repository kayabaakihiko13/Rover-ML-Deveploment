from ultralytics import YOLO

def test_simulate(model_path: str, data_path: str,device:str='cuda'):
    best_model = YOLO(model_path)
    best_model.val(data=data_path,
        split='test',
        batch=4,
        imgsz=640,
        device=device
    )

def saving_model(model_path: str):
    loaded_model = YOLO(model_path)
    loaded_model.export(format='onnx',device='cpu')

if __name__ == "__main__":
    best_model_path = r"runs\detect\train\weights\best.pt"
    data_path = r"data\data.yaml"
    export_path = r"runs\detect\train\weights\best.onnx"

    test_simulate(model_path=best_model_path, data_path=data_path,device='cuda')
    saving_model(best_model_path)
    test_simulate(model_path=best_model_path, data_path=data_path,device='cpu')
    
