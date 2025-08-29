#ifndef CAMERACALIBRATION_H
#define CAMERACALIBRATION_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <glm/glm.hpp>

class CameraCalibration
{
public:
    CameraCalibration();

    void setIntrinsicsMatrix(cv::Matx33d K);
    void setDistortionParameters(const std::vector<double>& newDists);


    cv::Matx33d getIntrinsicsMatrix() const;
    glm::mat3 getGlmIntrinsicsMatrix() const;
    double getFocalLengthX() const;
    double getFocalLengthY() const;
    double getPrincipalPointX() const;
    double getPrincipalPointY() const;

    const std::vector<double>& getDistortionParameters() const;
    int getHeight() const;
    void setHeight(int newHeight);

    int getWidth() const;
    void setWidth(int newWidth);

    bool isFishEye() const { return m_fishEye; }
    void setFishEye(bool fishEye) { m_fishEye = fishEye; }

    CameraCalibration getScaledCalibration(int newWidth, int newHeight);

    void saveCalibration(const std::string& file);
    bool loadCalibration(const std::string& file);
    bool loadCalibration(const double fx, const double fy, const double cx, const double cy, std::vector<double> dist, int width, int height);

    void save(std::ostream& out) const;
    void load(std::istream& in);

private:
    // Intrinsics
    cv::Matx33d m_K;

    // Distortion
    std::vector<double> m_dists;

    int m_width, m_height;

    bool m_fishEye;

};

#endif // CAMERACALIBRATION_H
