#include "Utils.h"

std::vector<cv::Point3d> Utils::transformCloud(const std::vector<cv::Point3d>& cloud, cv::Matx44d cloudToWorld)
{
    std::vector<cv::Point3d> transformedCloud = cloud;
    for (int i = 0; i < transformedCloud.size(); ++i)
        transformedCloud[i] = transformPoint(cloudToWorld, transformedCloud[i]);
    return transformedCloud;
}

cv::Point3d Utils::transformPoint(const cv::Matx44d& P, const cv::Point3d& pt)
{
    cv::Vec4d pt4(pt.x, pt.y, pt.z, 1.0);
    pt4 = P * pt4;
    return cv::Point3d(pt4[0], pt4[1], pt4[2]);
}

