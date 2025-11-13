#include "FeatureExtractor.h"
#include "PoseEstimator.h"

// argv[1] = "before.jpg" ← first image path
// argv[2] = "after.jpg" ← second image path


int main(int argc, char** argv) { 
    cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

    // 1) Create feature extractor
    FeatureExtractor extractor;

    // 2) Extract matched points
    MatchResult matches = extractor.extractAndMatch(img1, img2);

    // 3) Camera intrinsics (example placeholder)
    cv::Mat K = (cv::Mat_<double>(3,3) <<
                 500, 0, 320,
                 0, 500, 240,
                 0, 0, 1);

    // 4) Create pose estimator
    PoseEstimator poseEstimator(K);

    // 5) Compute 6-DoF pose
    PoseResult pose = poseEstimator.estimatePose(matches);

    // 6) Print results
    std::cout << "Rotation matrix R:\n" << pose.R << "\n\n";
    std::cout << "Translation vector T:\n" << pose.T << "\n\n";
    std::cout << "Roll: "  << pose.roll
              << " Pitch: " << pose.pitch
              << " Yaw: "   << pose.yaw  << std::endl;

    return 0;
}
