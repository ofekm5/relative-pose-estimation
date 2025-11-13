#pragma once
#include <opencv2/opencv.hpp>

struct MatchResult {
    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;
};