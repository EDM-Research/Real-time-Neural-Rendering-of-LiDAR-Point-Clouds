#include "Frustum.h"

cv::Vec3d Frustum::getDirForPixel(const cv::Matx44d& mvpInv, const cv::Vec3d& eye, const cv::Vec2i pixel) {
	float d = 1.0f;
	cv::Vec4d point(pixel[0] * d, pixel[1] * d, d, 1);
	cv::Vec4d pointDeproj4 = mvpInv * point;
	cv::Vec3d pointDeproj(pointDeproj4[0], pointDeproj4[1], pointDeproj4[2]);

	return cv::normalize(pointDeproj - eye);
}

Frustum::Frustum(const cv::Matx44d& K, const cv::Matx44d& view, int width, int heigth) {
	cv::Matx44d mvp = K * view;
	cv::Matx44d mvpInv = mvp.inv();
	cv::Matx44d viewInv = view.inv();

	cv::Vec3d eye(viewInv(0, 3), viewInv(1, 3), viewInv(2, 3));

	cv::Vec3d upperLeftDir = getDirForPixel(mvpInv, eye, cv::Vec2i(0, 0));
	cv::Vec3d bottomLeftDir = getDirForPixel(mvpInv, eye, cv::Vec2i(0, heigth));
	cv::Vec3d upperRightDir = getDirForPixel(mvpInv, eye, cv::Vec2i(width, 0));
	cv::Vec3d bottomRightDir = getDirForPixel(mvpInv, eye, cv::Vec2i(width, heigth));


	cv::Vec3d x(view(0, 0), view(1, 0), view(2, 0));
	cv::Vec3d y(view(0, 1), view(1, 1), view(2, 1));

	leftplane_n = -upperLeftDir.cross(bottomLeftDir);
	rightplane_n = upperRightDir.cross(bottomRightDir);
	bottomplane_n = -bottomLeftDir.cross(bottomRightDir);
	topplane_n = upperLeftDir.cross(upperRightDir);
	frontplane_n = x.cross(y);
	this->eye = eye;
}

bool Frustum::checkIfBoundingBoxIsInsideFrustum(const cv::Point3f& bbMin, const cv::Point3f& bbMax) {

	if (isInsideFrustum(cv::Vec3d(bbMin.x, bbMin.y, bbMin.z)))
		return true;
	if (isInsideFrustum(cv::Vec3d(bbMin.x, bbMin.y, bbMax.z)))
		return true;
	if (isInsideFrustum(cv::Vec3d(bbMin.x, bbMax.y, bbMin.z)))
		return true;
	if (isInsideFrustum(cv::Vec3d(bbMin.x, bbMax.y, bbMax.z)))
		return true;
	if (isInsideFrustum(cv::Vec3d(bbMax.x, bbMin.y, bbMin.z)))
		return true;
	if (isInsideFrustum(cv::Vec3d(bbMax.x, bbMin.y, bbMax.z)))
		return true;
	if (isInsideFrustum(cv::Vec3d(bbMax.x, bbMax.y, bbMin.z)))
		return true;
	if (isInsideFrustum(cv::Vec3d(bbMax.x, bbMax.y, bbMax.z)))
		return true;

	// check if center of block is inside
	//if (isInsideFrustum(cv::Vec3d(0.5*(bbMax.x- bbMin.x), 0.5 * (bbMax.y - bbMin.y), 0.5 * (bbMax.z - bbMin.z))))
	//    return true;

	return false;
}
