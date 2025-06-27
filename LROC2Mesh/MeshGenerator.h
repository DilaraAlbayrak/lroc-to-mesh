#pragma once
#include "DigitalTerrainModel.h"

class MeshGenerator
{
private:
    // Helper method to write all the vertex data to the file.
    void writeVertices(std::ofstream& file, const DigitalTerrainModel& dtm, double zScale, std::vector<int>& vertexMap) const;

    // Helper method to write all the face data, connecting the vertices.
    void writeFaces(std::ofstream& file, const DigitalTerrainModel& dtm, const std::vector<int>& vertexMap) const;

public:
	void generateMesh(const DigitalTerrainModel& dtm, const std::string& outputFilePath, double zScale = 1.0) const;
};

