import cv2, json, numpy as np
from tensorflow import keras

MODEL_PATH = "marker_model.keras"
NAMES_PATH = "class_names.json"
THRESH = 0.70          
CAM_INDEX = 0          
WINDOW_NAME = "Marker recognition"

model = keras.models.load_model(MODEL_PATH)
names = json.load(open(NAMES_PATH))

cap = cv2.VideoCapture(CAM_INDEX)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 960, 540)
print("Press q to quit, s to save a screenshot")
shot = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (224, 224)).astype("float32")[None]
    p = model.predict(x, verbose=0)[0]
    i = int(p.argmax())
    conf = float(p[i])

    if conf >= THRESH:
        label, color = names[i], (0, 200, 0)
    else:
        label, color = "unknown", (0, 0, 255)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(frame, f"{label}  {conf * 100:.0f}%", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s"):
        cv2.imwrite(f"demo_{shot}.png", frame)
        shot += 1

cap.release()
cv2.destroyAllWindows()
