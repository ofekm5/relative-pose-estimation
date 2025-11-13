# 🟦 **Partner Responsibilities **

## 👤 Partner A — FeatureExtractor

Implements:

* ORB detect + compute
* Matching + filtering
* Points extraction
  Delivers:
* `MatchResult` object

## 👤 Partner B — Docker Image + PoseEstimator

Implements:
1. Docker image for seamless build & deploy
2. C++ code:
* findEssentialMat
* recoverPose
* R → Euler
  Delivers:
* `PoseResult` object

---

# 🧩 How to Ensure Zero Conflicts

✔ Both partners write only inside their `.cpp` + `.h`
✔ A shared `include/` and `src/` folder
✔ One integration file (`main.cpp`) that does NOT change
✔ Agreed API between modules
