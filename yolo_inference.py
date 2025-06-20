from ultralytics import YOLO

model = YOLO('yolov8x')

print(f"Model info:\n{model.info}")

results = model.predict('input_videos/test_video.mp4', save=True)
print(results[0])
print(len(results))

print("==========================================")
for box in results[0].boxes:
    print(box)