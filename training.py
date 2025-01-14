from ultralytics import YOLO

model = YOLO('yolo11n.pt')

results = model.train(
   data='dt/data21.yaml',
   imgsz=192,
   epochs=400,
   batch=32,
   device="cpu",
   name='Hand_click2')