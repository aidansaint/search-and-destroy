from ultralytics import YOLO


def main():
    model = YOLO("models/yolo11n.pt")
    model.predict(source=0, classes=[0], show=True, save=False)


if __name__ == "__main__":
    main()
