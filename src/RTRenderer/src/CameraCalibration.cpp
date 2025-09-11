#include "CameraCalibration.h"
#include <fstream>
#include <iostream>

CameraCalibration::CameraCalibration()
{
    m_width = 640;
    m_height = 480;
    m_fishEye = false;
}

cv::Matx33d CameraCalibration::getIntrinsicsMatrix() const
{
    return m_K;
}

glm::mat3 CameraCalibration::getGlmIntrinsicsMatrix() const
{
    glm::mat3 result;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            result[row][col] = static_cast<float>(m_K(row, col)); 
            // glm is column-major: result[column][row]
        }
    }
    return result;
}

const std::vector<double>& CameraCalibration::getDistortionParameters()  const
{
    return m_dists;
}

int CameraCalibration::getHeight() const
{
    return m_height;
}

void CameraCalibration::setHeight(int newHeight)
{
    m_height = newHeight;
}

int CameraCalibration::getWidth() const
{
    return m_width;
}

void CameraCalibration::setWidth(int newWidth)
{
    m_width = newWidth;
}

CameraCalibration CameraCalibration::getScaledCalibration(int newWidth, int newHeight)
{
    CameraCalibration scaledCalibration;
    scaledCalibration.m_width = newWidth;
    scaledCalibration.m_height = newHeight;
    scaledCalibration.m_dists = m_dists;

    double scaleX = static_cast<double>(newWidth) / m_width;
    double scaleY = static_cast<double>(newHeight) / m_height;
    scaledCalibration.m_K = m_K;
    scaledCalibration.m_K(0, 0) *= scaleX;
    scaledCalibration.m_K(0, 2) *= scaleX;
    scaledCalibration.m_K(1, 1) *= scaleY;
    scaledCalibration.m_K(1, 2) *= scaleY;

    return scaledCalibration;
}

void CameraCalibration::saveCalibration(const std::string& file)
{
    std::ofstream ofs;
    ofs.open(file);

    if (ofs.is_open())
    {
        ofs << getWidth() << " " << getHeight() << std::endl;
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                ofs << m_K(r, c) << " ";
            }
            ofs << std::endl;
        }

        for (int d = 0; d < m_dists.size(); ++d)
            ofs << m_dists[d] << " ";
        ofs << std::endl;

        ofs << m_fishEye << std::endl;
    }
    else
    {
        std::cerr << "Failed to open file for saving calibration " << file << std::endl;
    }
}

bool CameraCalibration::loadCalibration(const std::string& file)
{
    std::cout << file << std::endl;
    // Special case: COLMAP-style cameras.txt
    if (file.size() >= 11 && file.substr(file.size() - 11) == "cameras.txt")
    {
        std::ifstream ifs(file);
        if (!ifs.is_open())
        {
            std::cerr << "Failed to open camera file for reading: " << file << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(ifs, line))
        {
            if (line.empty() || line[0] == '#')
                continue;

            std::istringstream iss(line);
            int cameraId;
            std::string model;
            float fx, fy, cx, cy;
            float k1, k2, p1, p2, k3;

            iss >> cameraId >> model >> m_width >> m_height;
            std::cout << m_width << "    " << m_height << std::endl;

            if (model != "OPENCV")
            {
                std::cerr << "Unsupported camera model: " << model << std::endl;
                return false;
            }

            iss >> fx >> fy >> cx >> cy >> k1 >> k2 >> p1 >> p2 >> k3;

            // Fill intrinsics
            m_K = cv::Matx33f::eye();
            m_K(0, 0) = fx;
            m_K(1, 1) = fy;
            m_K(0, 2) = cx;
            m_K(1, 2) = cy;

            // Fill distortion coefficients
            m_dists.resize(5, 0.0f);
            m_dists[0] = k1;
            m_dists[1] = k2;
            m_dists[2] = p1;
            m_dists[3] = p2;
            m_dists[4] = k3;

            m_fishEye = false;

            return true;
        }

        std::cerr << "No valid camera data found in cameras.txt" << std::endl;
        return false;
    }

    // Default: your original format
    std::ifstream ifs(file);
    if (!ifs.is_open())
    {
        std::cerr << "Failed to open calibration file for reading " << file << std::endl;
        return false;
    }

    ifs >> m_width;
    ifs >> m_height;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            ifs >> m_K(r, c);

    m_dists.resize(5);
    for (int d = 0; d < 5; ++d)
        ifs >> m_dists[d];

    ifs >> m_fishEye;

    return true;
}

bool CameraCalibration::loadCalibration(const double fx, const double fy, const double cx, const double cy, std::vector<double> dist, int width, int height)
{
    for(auto d: dist)
    {
        m_dists.push_back(d);
    }

    m_K = cv::Matx33d::eye();
    m_K(0,0) = fx;
    m_K(1,1) = fy;
    m_K(0, 2) = cx;
    m_K(1,2) = cy;

    m_width = width;
    m_height = height;

    m_fishEye = false;

    return true;
}

double CameraCalibration::getFocalLengthX() const
{
    return m_K(0, 0);
}

double CameraCalibration::getFocalLengthY() const
{
    return m_K(1, 1);
}

double CameraCalibration::getPrincipalPointX() const
{
    return m_K(0, 2);
}

double CameraCalibration::getPrincipalPointY() const
{
    return m_K(1, 2);
}

void CameraCalibration::setIntrinsicsMatrix(cv::Matx33d K)
{
    m_K = K;
}

void CameraCalibration::setDistortionParameters(const std::vector<double>& newDists)
{
    m_dists = newDists;
}

// Add these to CameraCalibration.cpp
void CameraCalibration::save(std::ostream& out) const {
    // Intrinsics
    out.write(reinterpret_cast<const char*>(&m_K), sizeof(cv::Matx33d));

    // Distortion parameters
    size_t dsize = m_dists.size();
    out.write(reinterpret_cast<const char*>(&dsize), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(m_dists.data()), dsize * sizeof(double));

    // Width, height, fisheye
    out.write(reinterpret_cast<const char*>(&m_width), sizeof(int));
    out.write(reinterpret_cast<const char*>(&m_height), sizeof(int));
    out.write(reinterpret_cast<const char*>(&m_fishEye), sizeof(bool));
}

void CameraCalibration::load(std::istream& in) {
    // Intrinsics
    in.read(reinterpret_cast<char*>(&m_K), sizeof(cv::Matx33d));

    // Distortion parameters
    size_t dsize;
    in.read(reinterpret_cast<char*>(&dsize), sizeof(size_t));
    m_dists.resize(dsize);
    in.read(reinterpret_cast<char*>(m_dists.data()), dsize * sizeof(double));

    // Width, height, fisheye
    in.read(reinterpret_cast<char*>(&m_width), sizeof(int));
    in.read(reinterpret_cast<char*>(&m_height), sizeof(int));
    in.read(reinterpret_cast<char*>(&m_fishEye), sizeof(bool));
}
