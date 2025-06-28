#pragma once
#include "DigitalTerrainModel.h"

class MeshGenerator
{
private:

    // Applies a simple 3x3 box blur. For each pixel, it averages the values
    // of the pixel and its 8 immediate neighbors.
    //void applySmoothing(const std::vector<float>& input, std::vector<float>& output, int width, int height, double noDataValue) const;

public:
	void generateMesh(const DigitalTerrainModel& dtm, const std::string& outputFilePath, int downscaleFactor = 1, double zScale = 1.0) const;
    void generateHeightmap(const DigitalTerrainModel& dtm, const std::string& outputFilePath) const;
};

