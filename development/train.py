import os
from ultralytics import YOLO

def train_model(model_path:str,data_path:str,config_label:str):
    if os.path.exists(model_path):
        if not model_path.lower().endswith('pt'):
            raise ValueError('Format File Model harus .pt')
    else:
        raise ValueError("Tidak Ada File nya")
    if os.path.exists('../runs'):
        os.rmdir('../runs')
    
    model = YOLO(model_path)
    model.train(data=data_path,cfg=config_label)


if __name__ == "__main__":
    data_path = "data\data.yaml"
    model_path = "model\yolo11s.pt"
    config_path = "development/simple_hyper.yaml"
    train_model(data_path=data_path,model_path=model_path,
                config_label=config_path) 
