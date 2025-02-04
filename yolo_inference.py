from ultralytics import YOLO

model = YOLO('models/best.pt')

print(f"Model info:\n{model.info}")

results = model.predict('input_videos/08fd33_4.mp4', save=True)
print(results[0])
print(len(results))

print("==========================================")
for box in results[0].boxes:
    print(box)