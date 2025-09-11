#include <opencv2/core.hpp>
#include <opencv2/core/quaternion.hpp>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
#include <iostream>
#include "RTRenderer/project_cloud.h"
#include "RTRenderer/cloudreader.h"
#include "RTRenderer/CameraCalibration.h"

struct TrajectoryEntry {
    int frame_id;
    cv::Matx44d pose;
    std::string filename;
};

TrajectoryEntry parseTrajectoryLine(const std::string& line) {
    std::istringstream iss(line);
    TrajectoryEntry entry;
    double qw, qx, qy, qz, tx, ty, tz;
    int camera_id;

    iss >> entry.frame_id >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> camera_id >> entry.filename;

    cv::Quatd q(qw, qx, qy, qz);
    cv::Matx33d R = q.toRotMat3x3();

    entry.pose = cv::Matx44d::eye();
    for (int i = 0; i < 3; ++i) {
        entry.pose(i, 0) = R(i, 0);
        entry.pose(i, 1) = R(i, 1);
        entry.pose(i, 2) = R(i, 2);
        entry.pose(i, 3) = (i == 0 ? tx : (i == 1 ? ty : tz));
    }

    return entry;
}

std::vector<TrajectoryEntry> readOrderedTrajectoryFile(const std::string& filename) {
    std::vector<TrajectoryEntry> traj_list;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        TrajectoryEntry entry = parseTrajectoryLine(line);
        traj_list.push_back(entry);
    }

    return traj_list;
}

int main(int argc, char**argv)
{
    if(argc < 3)
    {
        std::cerr << "Missing required parameter:" << std::endl;
        std::cerr << "render_trajectory pcl_path trajectory_path calibration_file" << std::endl;
        return -1;
    }

    std::filesystem::path path_to_cloud = std::filesystem::path(argv[1]);

    std::filesystem::path traj_path = std::filesystem::path(argv[2]);

    std::filesystem::path calib = std::filesystem::path(argv[3]);
    CameraCalibration calibration;
    calibration.loadCalibration(calib);
    
//    calibration = calibration.getScaledCalibration(960,720);

    // read point cloud (e57 or ply)
    auto grid = CloudReader::loadCloud(path_to_cloud, std::filesystem::path(getenv("HOME")) / ".pcl_cache"); // caches processed point cloud in ~/.pcl_cache, can change this to cache multiple point clouds
    std::shared_ptr<ProjectCloud> projector = std::make_shared<ProjectCloud>(grid, std::filesystem::path(getenv("HOME")) / ".pcl_cache" / "trt_1920.ts"); // <= replace with path to model file

    std::vector<TrajectoryEntry> trajectory = readOrderedTrajectoryFile(traj_path);

//    calibration.setHeight(1440);
//    calibration.setWidth(1440);

    for (const auto& entry : trajectory) {
        cv::Mat rgb = cv::Mat(cv::Size(calibration.getWidth(), calibration.getHeight()), CV_8UC3);
        cv::Mat depth = cv::Mat(cv::Size(calibration.getWidth(), calibration.getHeight()), CV_32F);
        std::cout << entry.pose << std::endl;
        projector->computeFilteredRGBD(calibration, entry.pose, & rgb, &depth);
        cv::imwrite(entry.filename, rgb);
    }
    return 0;
}
