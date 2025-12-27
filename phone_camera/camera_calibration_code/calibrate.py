import cv2
import numpy as np
import glob
import os
from collections import Counter

# =========================================================
# הגדרות
# =========================================================
IMAGE_GLOB = "/home/orr/university_projects/relative-pose-estimation/chess_calibration/*.jpeg"

# 8x8 משבצות => 7x7 פינות פנימיות
CHESSBOARD_SIZE = (7, 7)

# גודל משבצת: 4 ס"מ
SQUARE_SIZE = 4.0

# סינון תמונות לפי שגיאה (פיקסלים)
PER_IMAGE_ERR_THRESH = 1

# מינימום תמונות להשאיר (אם הסף משאיר פחות - ניקח את הטובות ביותר)
KEEP_MIN = 10

# האם להשתמש רק ברזולוציה הנפוצה ביותר (מומלץ!)
USE_MOST_COMMON_SIZE_ONLY = True

# =========================================================
# יצירת נקודות בעולם (XYZ) ללוח השחמט
# =========================================================
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# =========================================================
# טעינת קבצים
# =========================================================
images = sorted(glob.glob(IMAGE_GLOB))
if not images:
    raise RuntimeError(f"לא נמצאו תמונות לפי: {IMAGE_GLOB}")

print(f"נמצאו {len(images)} תמונות")

# =========================================================
# בדיקת רזולוציות
# =========================================================
sizes = []
for f in images:
    im = cv2.imread(f)
    if im is None:
        continue
    h, w = im.shape[:2]
    sizes.append((w, h))

if not sizes:
    raise RuntimeError("לא הצלחתי לקרוא אף תמונה (cv2.imread נכשל).")

cnt = Counter(sizes)
print("\n=== Unique image sizes ===")
for sz, c in cnt.most_common():
    print(f"{sz} -> {c}")

# =========================================================
# סינון לרזולוציה הנפוצה ביותר (מומלץ מאוד)
# =========================================================
if USE_MOST_COMMON_SIZE_ONLY:
    most_common_size = cnt.most_common(1)[0][0]
    filtered = []
    for f in images:
        im = cv2.imread(f)
        if im is None:
            continue
        h, w = im.shape[:2]
        if (w, h) == most_common_size:
            filtered.append(f)
    images = filtered
    print(f"\nUsing only images of size {most_common_size}, count={len(images)}")

# =========================================================
# מציאת פינות
# =========================================================
objpoints = []
imgpoints = []
used_files = []
img_size = None

print("\n=== Detecting chessboard corners ===")
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"⚠️ לא הצלחתי לקרוא: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_size = gray.shape[::-1]  # (w,h)

    ret, corners = cv2.findChessboardCorners(
        gray,
        CHESSBOARD_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if not ret:
        print(f"❌ לא זוהה לוח: {os.path.basename(fname)}")
        continue

    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )

    objpoints.append(objp)
    imgpoints.append(corners)
    used_files.append(fname)

print(f"\n✅ זוהה לוח בהצלחה ב-{len(used_files)} תמונות")
if len(used_files) < 6:
    raise RuntimeError("מעט מדי תמונות עם זיהוי לוח. צריך לפחות ~6, מומלץ 10-15+.")

# =========================================================
# קליברציה ראשונית
# =========================================================
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_size, None, None
)

print("\n=== Initial calibration ===")
print("Global reprojection error:", ret)
print("K:\n", K)
print("dist:\n", dist)

# =========================================================
# reprojection error לכל תמונה
# =========================================================
per_image = []
for i in range(len(used_files)):
    projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
    err = cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(projected)
    per_image.append((used_files[i], float(err)))

per_image_sorted = sorted(per_image, key=lambda x: x[1])

print("\n=== Per-image reprojection errors (low=good) ===")
for f, e in per_image_sorted:
    print(f"{os.path.basename(f):35s} err={e:.3f}")

# שמירת CSV
with open("per_image_errors.csv", "w", encoding="utf-8") as fp:
    fp.write("filename,error\n")
    for f, e in per_image_sorted:
        fp.write(f"{os.path.basename(f)},{e:.6f}\n")
print("\nנשמר: per_image_errors.csv")

# =========================================================
# בחירת תמונות טובות
# =========================================================
good = [(f, e) for (f, e) in per_image_sorted if e <= PER_IMAGE_ERR_THRESH]

if len(good) < KEEP_MIN:
    good = per_image_sorted[:min(KEEP_MIN, len(per_image_sorted))]
    print(f"\n⚠️ לפי הסף נשארו מעט מדי, אז אני שומר את {len(good)} הטובות ביותר במקום.")
else:
    print(f"\n✅ לפי הסף נשארו {len(good)} תמונות טובות (<= {PER_IMAGE_ERR_THRESH})")

good_files = [f for f, _ in good]

# שמירת רשימת קבצים טובים
with open("good_images.txt", "w", encoding="utf-8") as fp:
    for f in good_files:
        fp.write(f + "\n")
print("נשמר: good_images.txt")

# =========================================================
# קליברציה מחדש רק עם הטובות
# =========================================================
objpoints2 = []
imgpoints2 = []
img_size2 = None

print("\n=== Re-calibrating using good images only ===")
for fname in good_files:
    img = cv2.imread(fname)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_size2 = gray.shape[::-1]

    ret2, corners2 = cv2.findChessboardCorners(
        gray,
        CHESSBOARD_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not ret2:
        continue

    corners2 = cv2.cornerSubPix(
        gray,
        corners2,
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )

    objpoints2.append(objp)
    imgpoints2.append(corners2)

retF, KF, distF, rvecsF, tvecsF = cv2.calibrateCamera(
    objpoints2, imgpoints2, img_size2, None, None
)

print("\n=== Filtered calibration results ===")
print("Global reprojection error:", retF)
print("K:\n", KF)
print("dist:\n", distF)

np.savez("calibration_filtered.npz", K=KF, dist=distF)
print("\nנשמר: calibration_filtered.npz (K, dist)")
