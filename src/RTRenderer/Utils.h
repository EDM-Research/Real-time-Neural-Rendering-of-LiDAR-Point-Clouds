#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>

namespace Utils
{
    cv::Point3d transformPoint(const cv::Matx44d& P, const cv::Point3d& pt);

    std::vector<cv::Point3d> transformCloud(const std::vector<cv::Point3d>& cloud, cv::Matx44d cloudToWorld);
};

#endif // UTILS_H
