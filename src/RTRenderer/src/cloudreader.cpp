#include "cloudreader.h"
#include "PointCloudReader.h"
#include "Utils.h"
#include "project_cloud.h"
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"

void computeGrid(std::vector<cv::Point3f> cloud, std::vector<cv::Vec3b> colors, std::unordered_map<int, OctreeGrid::Block>& grid, int& numX, int& numY, int& numZ)
{
    float block_size = 0.25f; // in meters

    cv::Point3f bbMin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    cv::Point3f bbMax(std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min());
    for (const cv::Point3f& pt : cloud) {
        if (pt.x < bbMin.x)
            bbMin.x = pt.x;
        if (pt.y < bbMin.y)
            bbMin.y = pt.y;
        if (pt.z < bbMin.z)
            bbMin.z = pt.z;

        if (pt.x > bbMax.x)
            bbMax.x = pt.x;
        if (pt.y > bbMax.y)
            bbMax.y = pt.y;
        if (pt.z > bbMax.z)
            bbMax.z = pt.z;
    }

    // rounding to meters
    bbMax.x = ::ceilf(bbMax.x);
    bbMax.y = ::ceilf(bbMax.y);
    bbMax.z = ::ceilf(bbMax.z);
    bbMin.x = ::floorf(bbMin.x);
    bbMin.y = ::floorf(bbMin.y);
    bbMin.z = ::floorf(bbMin.z);

    // create 3D grid
    int numBlocks_x = int((bbMax.x - bbMin.x) / block_size);
    int numBlocks_y = int((bbMax.y - bbMin.y) / block_size);
    int numBlocks_z = int((bbMax.z - bbMin.z) / block_size);
    numX = numBlocks_x;
    numY = numBlocks_y;
    numZ = numBlocks_z;

    // assign points to blocks
    for (size_t i = 0; i < cloud.size(); ++i) {
        cv::Point3f pt = cloud[i];
        cv::Vec3b color = colors[i];

        int x = std::floor((pt.x - bbMin.x) / (bbMax.x - bbMin.x) * numBlocks_x);
        int y = std::floor((pt.y - bbMin.y) / (bbMax.y - bbMin.y) * numBlocks_y);
        int z = std::floor((pt.z - bbMin.z) / (bbMax.z - bbMin.z) * numBlocks_z);

        if (x < 0 || y < 0 || z < 0 || x >= numBlocks_x || y >= numBlocks_y || z >= numBlocks_z)
            std::cerr << "out of bounds: " << x << ", " << y << ", " << z << std::endl;

        grid[OctreeGrid::encodeKey(x, y, z, numBlocks_x, numBlocks_y, numBlocks_z)].positions.emplace_back(pt);
        grid[OctreeGrid::encodeKey(x, y, z, numBlocks_x, numBlocks_y, numBlocks_z)].colors.emplace_back(color);
    }

    for (auto& block_pair : grid) {

        int x, y, z;
        OctreeGrid::decodeKey(block_pair.first, x, y, z, numBlocks_x, numBlocks_y, numBlocks_z);


        float bbSize_x = (bbMax.x - bbMin.x) / numBlocks_x;
        block_pair.second.bbMin.x = bbMin.x + x * bbSize_x;
        block_pair.second.bbMax.x = bbMin.x + (x + 1) * bbSize_x;

        float bbSize_y = (bbMax.y - bbMin.y) / numBlocks_y;
        block_pair.second.bbMin.y = bbMin.y + y * bbSize_y;
        block_pair.second.bbMax.y = bbMin.y + (y + 1) * bbSize_y;

        float bbSize_z = (bbMax.z - bbMin.z) / numBlocks_z;
        block_pair.second.bbMin.z = bbMin.z + z * bbSize_z;
        block_pair.second.bbMax.z = bbMin.z + (z + 1) * bbSize_z;

    }
    return;
}

void loadE57(const std::filesystem::path &file_name, std::unordered_map<int, OctreeGrid::Block>& grid, int &numX, int& numY, int& numZ)
{
    PointCloudReader reader(file_name.string());

    int numOfClouds = reader.getNumberOfClouds();
    cv::Matx44d scanToWorld;

    std::vector<cv::Point3f> cloud;
    std::vector<cv::Vec3b> colors;

    std::cout << "Reading " << numOfClouds << " clouds" << std::endl;
    for (unsigned int i = 0; i < numOfClouds; ++i) {
        std::cout << "Reading point cloud " << i << std::endl;
        std::vector<cv::Point3d> localCloud;
        std::vector<cv::Vec3b> localColors;
        reader.getScanCloud(i, scanToWorld, localCloud, localColors, 0);

        localCloud = Utils::transformCloud(localCloud, scanToWorld);

        colors.insert(colors.end(), localColors.begin(), localColors.end());

        for (int pointIdx = 0; pointIdx < localCloud.size(); ++pointIdx)
        {
            cv::Point3f pos;
            pos.x = (float)localCloud[pointIdx].x;
            pos.y = (float)localCloud[pointIdx].y;
            pos.z = (float)localCloud[pointIdx].z;

            cloud.emplace_back(pos);
        }
    }

    std::cout << "Found " << cloud.size() << " points" << std::endl;

    computeGrid(cloud, colors, grid, numX, numY, numZ);

}

void loadPLY(const std::filesystem::path &file_name, std::unordered_map<int, OctreeGrid::Block>& grid, int &numX, int& numY, int& numZ)
{
    std::vector<cv::Point3f> cloud;
    std::vector<cv::Vec3b> colors;

    try {
        std::ifstream ss(file_name, std::ios::binary);
        if (!ss.is_open()) {
            std::cerr << "Failed to open file: " << file_name << std::endl;
            exit(-1);
        }

        tinyply::PlyFile file;
        file.parse_header(ss);

        std::shared_ptr<tinyply::PlyData> vertices, rgb;

        try {
            vertices = file.request_properties_from_element("vertex", { "x", "y", "z" });
        } catch (const std::exception& e) {
            std::cerr << "Missing vertex positions: " << e.what() << std::endl;
            exit(-1);
        }

        try {
            rgb = file.request_properties_from_element("vertex", { "red", "green", "blue" });
        } catch (...) {
            // Color is optional
        }

        file.read(ss);

        // Copy point data
        const size_t numVertices = vertices->count;
        const float* verts = reinterpret_cast<const float*>(vertices->buffer.get());
        cloud.resize(numVertices);
        for (size_t i = 0; i < numVertices; ++i) {
            cloud[i] = cv::Point3f(verts[3 * i + 0], verts[3 * i + 1], verts[3 * i + 2]);
        }

        // Copy color data if available
        colors.clear();
        if (rgb) {
            const uint8_t* cols = reinterpret_cast<const uint8_t*>(rgb->buffer.get());
            colors.resize(numVertices);
            for (size_t i = 0; i < numVertices; ++i) {
                colors[i] = cv::Vec3b(cols[3 * i + 2], cols[3 * i + 1], cols[3 * i + 0]); // OpenCV uses BGR
            }
        }

        computeGrid(cloud, colors, grid, numX, numY, numZ);
    } catch (const std::exception& e) {
        std::cerr << "Exception reading PLY: " << e.what() << std::endl;
        exit(-1);
    }
}


std::unordered_map<int, OctreeGrid::Block> CloudReader::loadCloud(const std::filesystem::path &file_name, const std::filesystem::path& caching_dir)
{
    if(caching_dir.string() != "")
    {
        if(std::filesystem::exists(caching_dir/ "pcd.oct"))
        {
            std::unordered_map<int, OctreeGrid::Block> grid;
            int numBlocks_x, numBlocks_y, numBlocks_z;
            OctreeGrid::readOctreeBinary((caching_dir / "pcd.oct").string(), grid, numBlocks_x, numBlocks_y, numBlocks_z);
            return grid;
        }
    }

    std::unordered_map<int, OctreeGrid::Block> grid;
    int numBlocks_x, numBlocks_y, numBlocks_z;

    if(file_name.extension() == ".e57")
    {
        loadE57(file_name, grid, numBlocks_x, numBlocks_y, numBlocks_z);
    }
    else if(file_name.extension() == ".ply")
    {
        loadPLY(file_name, grid, numBlocks_x, numBlocks_y, numBlocks_z);
    }
    else
    {
        std::cerr << "File extension " << file_name.extension() << " not supported. Only e57 and ply are supported at this moment." << std::endl;
        exit(-1);
    }

    if(caching_dir.string() != "")
    {
        std::filesystem::create_directories(caching_dir);
        OctreeGrid::writeOctreeBinary((caching_dir / "pcd.oct").string(), grid, numBlocks_x, numBlocks_y, numBlocks_z);
    }
    return grid;
}


void CloudReader::loadCubemaps(const std::filesystem::path &file_name, std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &depths, std::vector<cv::Matx44d> &w2cam, std::vector<CameraCalibration> &calibs)
{
    if(file_name.extension() != ".e57")
    {
        std::cerr << "Not supported file extension" << std::endl;
        return;
    }

    PointCloudReader reader(file_name.string());

    int numberOfImages = reader.getNumberOfImages();
    std::cout << numberOfImages << std::endl;

    auto grid = loadCloud(file_name);
    ProjectCloud projector{grid};

    for(int i = 0; i < numberOfImages; i++)
    {
        std::cout << i << std::endl;
        cv::Matx44d pose; cv::Matx33d intrinsics;
        cv::Mat img = reader.getImage(i, pose, intrinsics);

        CameraCalibration calib;
        calib.setIntrinsicsMatrix(intrinsics);
        calib.setWidth(img.cols);
        calib.setHeight(img.rows);
        cv::Mat depth(img.size(), CV_32F);
        projector.computeRGBD(calib, pose, nullptr, &depth);

        imgs.push_back(img);
        depths.push_back(depth);
        w2cam.push_back(pose);
        calibs.push_back(calib);
    }

}
