#include "PointCloudReader.h"
using namespace e57;


PointCloudReader::PointCloudReader(const std::string& filename)
{
    ReaderOptions readerOptions;
    _reader = new Reader(filename, readerOptions);

}

int PointCloudReader::getNumberOfImages()
{
    return _reader->GetImage2DCount();
}

cv::Mat PointCloudReader::getImage(int index, cv::Matx44d& pose, cv::Matx33d& intrinsics)
{
    Image2D image;
    if (_reader->ReadImage2D(index, image))
    {
        pose = obtainCameraExtrinsics(image.pose).inv();

        cv::Matx33d K = cv::Matx33d::eye();
        K(0, 0) = (image.pinholeRepresentation.focalLength / image.pinholeRepresentation.pixelWidth);
        K(1, 1) = (image.pinholeRepresentation.focalLength / image.pinholeRepresentation.pixelHeight);

        K(0, 2) = image.pinholeRepresentation.principalPointX;
        K(1, 2) = image.pinholeRepresentation.principalPointY;
        intrinsics = K;

        Image2DProjection imageProjection;
        Image2DType imageType;
        int64_t imageWidth, imageHeight, imageSize;
        Image2DType imageMaskType, imageVisualType;
        if (_reader->GetImage2DSizes(index, imageProjection, imageType, imageWidth, imageHeight, imageSize,
            imageMaskType, imageVisualType))
        {
            if (imageType != Image2DType::E57_NO_IMAGE)
            {
                std::vector<uint8_t> imageBuffer(imageSize);
                _reader->ReadImage2DData(index, imageProjection, imageType, &imageBuffer[0], 0, imageSize);

                cv::Mat image = cv::imdecode(imageBuffer, cv::IMREAD_COLOR);
                return image;
            }
        }
    }
    return cv::Mat();
}

cv::Matx33d PointCloudReader::quatToRot3x3(Quaternion q)
{
    double a = q.w, b = q.x, c = q.y, d = q.z;
    double length = sqrt(a * a + b * b + c * c + d * d);
    a /= length;
    b /= length;
    c /= length;
    d /= length;

    cv::Matx33d R{
        1 - 2 * (c * c + d * d), 2 * (b * c - a * d)    , 2 * (b * d + a * c),
                2 * (b * c + a * d)    , 1 - 2 * (b * b + d * d), 2 * (c * d - a * b),
                2 * (b * d - a * c)    , 2 * (c * d + a * b)    , 1 - 2 * (b * b + c * c)
    };

    return R;
}

cv::Matx44d PointCloudReader::obtainCameraExtrinsics(RigidBodyTransform t)
{
    cv::Matx33d R = quatToRot3x3(t.rotation);
    cv::Matx44d P = cv::Matx44d::eye();

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            P(i, j) = R(i, j);

    P(0, 3) = t.translation.x;
    P(1, 3) = t.translation.y;
    P(2, 3) = t.translation.z;

    cv::Matx44d I = cv::Matx44d::eye();
    I(2, 2) = -1;
    P = P * I;
    I(2, 2) = 1;
    I(1, 1) = -1;
    P = P * I;


    return P;
}

int PointCloudReader::getNumberOfClouds()
{
    return _reader->GetData3DCount();
}

cv::Matx44d PointCloudReader::transformCloudToWorld(RigidBodyTransform t)
{
    cv::Matx33d R = quatToRot3x3(t.rotation);
    cv::Matx44d P = cv::Matx44d::eye();

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            P(i, j) = R(i, j);

    P(0, 3) = t.translation.x;
    P(1, 3) = t.translation.y;
    P(2, 3) = t.translation.z;

    return P;
}

void PointCloudReader::getScanCloud(int scanIndex, cv::Matx44d& scanToWorldPose, std::vector<cv::Point3d>& cloud, std::vector<cv::Vec3b>& colors, int skip)
{
    int numberOfClouds = _reader->GetData3DCount();

    Data3DPointsData_d buffers;
    e57::Data3D		scanHeader;
    _reader->ReadData3D(scanIndex, scanHeader);


    //Get the Size of the Scan
    int64_t nColumn = 0;
    int64_t nRow = 0;
    int64_t nPointsSize = 0;	//Number of points
    int64_t nGroupsSize = 0;	//Number of groups
    int64_t nCountSize = 0;		//Number of points per group
    bool	bColumnIndex = false; //indicates that idElementName is "columnIndex"

    _reader->GetData3DSizes(scanIndex, nRow, nColumn, nPointsSize, nGroupsSize, nCountSize, bColumnIndex);

    int64_t nSize = nRow;
    if (nSize == 0) nSize = 10240;	// choose a chunk size

    //Setup buffers
    buffers.cartesianInvalidState = NULL;
    if (scanHeader.pointFields.cartesianInvalidStateField)
        buffers.cartesianInvalidState = new int8_t[nSize];

    //Setup Points Buffers
    buffers.cartesianX = NULL;
    if (scanHeader.pointFields.cartesianXField)
        buffers.cartesianX = new double[nSize];

    buffers.cartesianY = NULL;
    if (scanHeader.pointFields.cartesianYField)
        buffers.cartesianY = new double[nSize];

    buffers.cartesianZ = NULL;
    if (scanHeader.pointFields.cartesianZField)
        buffers.cartesianZ = new double[nSize];

    //Setup buffers spherical
    buffers.sphericalInvalidState = NULL;
    if (scanHeader.pointFields.sphericalInvalidStateField)
        buffers.sphericalInvalidState = new int8_t[nSize];

    //Setup Points Buffers
    buffers.sphericalAzimuth = NULL;
    if (scanHeader.pointFields.sphericalAzimuthField)
        buffers.sphericalAzimuth = new double[nSize];

    buffers.sphericalElevation = NULL;
    if (scanHeader.pointFields.sphericalElevationField)
        buffers.sphericalElevation = new double[nSize];

    buffers.sphericalRange = NULL;
    if (scanHeader.pointFields.sphericalRangeField)
        buffers.sphericalRange = new double[nSize];

    //Setup intensity buffers if present
    buffers.intensity = NULL;
    bool		bIntensity = false;
    double		intRange = 0;
    double		intOffset = 0;
    if (scanHeader.pointFields.intensityField)
    {
        bIntensity = true;
        buffers.intensity = new double[nSize];

        intRange = scanHeader.intensityLimits.intensityMaximum - scanHeader.intensityLimits.intensityMinimum;
        intOffset = scanHeader.intensityLimits.intensityMinimum;
    }


    //Setup color buffers if present
    buffers.colorRed = NULL;
    buffers.colorGreen = NULL;
    buffers.colorBlue = NULL;
    bool		bColor = false;
    double		colorRedRange = 1;
    double		colorRedOffset = 0;
    double		colorGreenRange = 1;
    double		colorGreenOffset = 0;
    double		colorBlueRange = 1;
    double		colorBlueOffset = 0;
    if (scanHeader.pointFields.colorRedField)
    {
        bColor = true;
        buffers.colorRed = new uint16_t[nSize];
        buffers.colorGreen = new uint16_t[nSize];
        buffers.colorBlue = new uint16_t[nSize];

        colorRedRange = scanHeader.colorLimits.colorRedMaximum - scanHeader.colorLimits.colorRedMinimum;
        colorRedOffset = scanHeader.colorLimits.colorRedMinimum;
        colorGreenRange = scanHeader.colorLimits.colorGreenMaximum - scanHeader.colorLimits.colorGreenMinimum;
        colorGreenOffset = scanHeader.colorLimits.colorGreenMinimum;
        colorBlueRange = scanHeader.colorLimits.colorBlueMaximum - scanHeader.colorLimits.colorBlueMinimum;
        colorBlueOffset = scanHeader.colorLimits.colorBlueMinimum;
    }

    //Setup the GroupByLine buffers information
    int64_t* idElementValue = NULL;
    int64_t* startPointIndex = NULL;
    int64_t* pointCount = NULL;
    if (nGroupsSize > 0)
    {
        idElementValue = new int64_t[nGroupsSize];
        startPointIndex = new int64_t[nGroupsSize];
        pointCount = new int64_t[nGroupsSize];
        if (!_reader->ReadData3DGroupsData(scanIndex, nGroupsSize, idElementValue, startPointIndex, pointCount))
            nGroupsSize = 0;
    }

    //Setup row/column index information
    buffers.rowIndex = NULL;
    buffers.columnIndex = NULL;
    if (scanHeader.pointFields.rowIndexField)
        buffers.rowIndex = new int32_t[nSize];
    if (scanHeader.pointFields.columnIndexField)
        buffers.columnIndex = new int32_t[nRow];


    //Get dataReader object
    e57::CompressedVectorReader dataReader = _reader->SetUpData3DPointsData(scanIndex, nSize, buffers);
    scanToWorldPose = transformCloudToWorld(scanHeader.pose);

    //Read the point data
    int64_t		count = 0;
    unsigned	size = 0;
    int			col = 0;
    int			row = 0;


    cloud.clear();
    colors.clear();
    cloud.reserve(nPointsSize);
    colors.reserve(nPointsSize);

    while ((size = dataReader.read()) != 0)
    {
        //Use the data
        for (long i = 0; i < size; i += (skip == 0) ? 1 : ((rand() % skip) + 1))
        {
            if (buffers.cartesianInvalidState && buffers.cartesianInvalidState[i] == 0)
            {
                cv::Point3d pt(buffers.cartesianX[i], buffers.cartesianY[i], buffers.cartesianZ[i]);
                int red = ((buffers.colorRed[i] - colorRedOffset) * 255) / colorRedRange;
                int green = ((buffers.colorGreen[i] - colorGreenOffset) * 255) / colorBlueRange;
                int blue = ((buffers.colorBlue[i] - colorBlueOffset) * 255) / colorBlueRange;

                cv::Vec3b color(blue, green, red);
                cloud.push_back(pt);
                colors.push_back(color);
            }
            else if (buffers.sphericalInvalidState && buffers.sphericalInvalidState[i] == 0)
            {
                double azimuth = buffers.sphericalAzimuth[i];
                double elevation = buffers.sphericalElevation[i];
                double range = buffers.sphericalRange[i];

                double cartesianX = range * cos(azimuth) * cos(elevation);
                double cartesianY = range * sin(azimuth) * cos(elevation);
                double cartesianZ = range * sin(elevation);

                cv::Point3d pt(cartesianX, cartesianY, cartesianZ);
                int red = ((buffers.colorRed[i] - colorRedOffset) * 255) / colorRedRange;
                int green = ((buffers.colorGreen[i] - colorGreenOffset) * 255) / colorBlueRange;
                int blue = ((buffers.colorBlue[i] - colorBlueOffset) * 255) / colorBlueRange;

                cv::Vec3b color(blue, green, red);
                cloud.push_back(pt);
                colors.push_back(color);
            }

        }
    }

    //Close and clean up
    dataReader.close();
}
