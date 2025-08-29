#pragma once

class Config
{
public:
	inline static float resizeParam = 1;
	inline static float decimateParam = 1;

	inline static const int depthRescaleDepth = 4;
	inline static const float filterStrength = 1.025;
	inline static const float gradientFilter = 0.03;
};