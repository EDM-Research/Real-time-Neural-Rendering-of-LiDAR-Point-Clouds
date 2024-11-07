#pragma once
#ifndef POINTCLOUDREADER_H
#define POINTCLOUDREADER_H

#include <E57Format/E57SimpleReader.h>
#include <opencv2/opencv.hpp>

class PointCloudReader
{
public:
    PointCloudReader(const std::string& filename);


    int getNumberOfClouds();
    int getNumberOfImages();
    cv::Mat getImage(int index, cv::Matx44d& pose, cv::Matx33d& intrinsics);

    void getScanCloud(int scanIndex, cv::Matx44d& scanToWorldPose, std::vector<cv::Point3d>& cloud, std::vector<cv::Vec3b>& colors, int skip = 0);
private:
    e57::Reader* _reader;

    cv::Matx44d obtainCameraExtrinsics(e57::RigidBodyTransform t);
    cv::Matx33d quatToRot3x3(e57::Quaternion q);
    cv::Matx44d transformCloudToWorld(e57::RigidBodyTransform t);
};



#endif // POINTCLOUDREADER_H
