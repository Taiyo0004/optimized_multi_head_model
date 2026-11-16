import cv2, numpy as np
from ultralytics import YOLO

IMG = r"step7_synthetic\images\val\val_000.jpg"
WEIGHTS = r"runs\multi\step7_fix_white\weights\best.pt"

res = YOLO(WEIGHTS, task="multi").predict(source=IMG, imgsz=320, verbose=False)
mask = res[1][0].cpu().numpy()          # per pixel: 0 = background, 1 = solid, 2 = dashed
img = cv2.imread(IMG)

solid = img.copy();  solid[mask == 1] = (0, 255, 0)    # green
dashed = img.copy(); dashed[mask == 2] = (0, 0, 255)   # red
for im, text in ((solid, "SOLID"), (dashed, "DASHED")):
    cv2.putText(im, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.imwrite("lanes_solid_vs_dashed.jpg", np.hstack([solid, dashed]))
print("solid pixels:", int((mask == 1).sum()), " dashed pixels:", int((mask == 2).sum()))