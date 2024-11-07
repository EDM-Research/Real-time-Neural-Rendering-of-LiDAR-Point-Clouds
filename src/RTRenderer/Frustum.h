#pragma once

#include <opencv2/opencv.hpp>

class Frustum {
public: 
    Frustum(const cv::Matx44d& K, const cv::Matx44d& view, int width, int heigth);
    bool checkIfBoundingBoxIsInsideFrustum(const cv::Point3f& bbMin, const cv::Point3f& bbMax);

    cv::Vec3d eye;

private:
    cv::Vec3d leftplane_n;
    cv::Vec3d rightplane_n;
    cv::Vec3d bottomplane_n;
    cv::Vec3d topplane_n;
    cv::Vec3d frontplane_n;

    cv::Vec3d getDirForPixel(const cv::Matx44d& mvpInv, const cv::Vec3d& eye, const cv::Vec2i pixel);

    bool isInsideFrustum(const cv::Vec3d& point) const {
        if ((point - eye).dot(leftplane_n) < 0.0f)
            return false;
        if ((point - eye).dot(rightplane_n) < 0.0f)
            return false;
        if ((point - eye).dot(bottomplane_n) < 0.0f)
            return false;
        if ((point - eye).dot(topplane_n) < 0.0f)
            return false;
        /*       if ((point - eye).dot(frontplane_n) < 0.0f)
                    return false;*/

        return true;
    }
};
