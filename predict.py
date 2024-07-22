from ultralytics import YOLO
def prediction(image_path):
    model = YOLO('runs/classify/train/weights/best.pt')
    results = model.predict(image_path)
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    pred = names_dict[probs.index(max(probs))]
    return pred


if __name__ == "__main__":
    image_path = "data/val/Rosado/rosado_77_PNG.rf.2023b8647a9fc8b5c1b253fa16027e71.jpg"
    prediction(image_path)