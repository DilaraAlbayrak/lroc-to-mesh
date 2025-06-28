#include "MeshGenerator.h"
#include <iostream>
#include <fstream>

// for openmp
#include <sstream> 
#include <omp.h>  

#include <chrono>
#include <iomanip> // Required for std::setprecision

// for heightmap generation
#include <algorithm> // for std::min_element, std::max_element
#include <cmath> // for std::round
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct Vector3
{
	double x = 0.0, y = 0.0, z = 0.0;

	Vector3& operator+=(const Vector3& other) {
		x += other.x;
		y += other.y;
		z += other.z;
		return *this;
	}

	void normalize()
	{
		double length = std::sqrt(x * x + y * y + z * z);
		if (length > 1e-6) { // avoid division by zero
			x /= length;
			y /= length;
			z /= length;
		}
	}
};

Vector3 cross(const Vector3& a, const Vector3& b)
{
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	};
}

// Overload the - operator for Vector3
Vector3 operator-(const Vector3& a, const Vector3& b) {
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}

void MeshGenerator::generateMesh(const DigitalTerrainModel& dtm, const std::string& outputFilePath, int downscaleFactor, double zScale) const
{
    if (downscaleFactor < 1) {
        std::cout << ">>>> warning: downscaleFactor cannot be less than 1. Setting to 1 (no downscaling).\n";
        downscaleFactor = 1;
    }

    auto total_start = std::chrono::high_resolution_clock::now();
    std::cout << ">>>> generating mesh for DTM with normals (Downscale: " << downscaleFactor << "x, OpenMP: enabled)" << std::endl;

    const int width = dtm.getWidth();
    const int height = dtm.getHeight();

    // --- 1. Generate vertex data in memory ---
    std::cout << "\n>>>> [1/5] generating vertices in memory...\n";
    std::vector<Vector3> vertices;
    std::vector<int> vertexMap(static_cast<size_t>(width) * height, 0);

    const double* geoTransform = dtm.getGeoTransform();
    const float noDataValue = dtm.getNoDataValue();
    std::vector<float> scanlineBuffer(width);

    for (int y = 0; y < height; y += downscaleFactor) {
        if (!dtm.readScanline(y, scanlineBuffer)) continue;
        for (int x = 0; x < width; x += downscaleFactor) {
            float elevation = scanlineBuffer[x];
            if (elevation == noDataValue) continue;

            Vector3 pos;
            pos.x = geoTransform[0] + x * geoTransform[1] + y * geoTransform[2];
            pos.y = geoTransform[3] + x * geoTransform[4] + y * geoTransform[5];
            pos.z = -elevation * zScale;

            vertices.push_back(pos);
            vertexMap[static_cast<size_t>(y) * width + x] = static_cast<int>(vertices.size());
        }
    }
    std::cout << ">>>> generated " << vertices.size() << " vertices.\n";

    // --- Center the Model to Origin  ---
    if (!vertices.empty()) {
        std::cout << "\n>>>> [2/5] centering the model to origin...\n";
        Vector3 min_bound = vertices[0];
        Vector3 max_bound = vertices[0];

        for (size_t i = 1; i < vertices.size(); ++i) {
            min_bound.x = std::min(min_bound.x, vertices[i].x);
            min_bound.y = std::min(min_bound.y, vertices[i].y);
            min_bound.z = std::min(min_bound.z, vertices[i].z);
            max_bound.x = std::max(max_bound.x, vertices[i].x);
            max_bound.y = std::max(max_bound.y, vertices[i].y);
            max_bound.z = std::max(max_bound.z, vertices[i].z);
        }

        Vector3 center;
        center.x = min_bound.x + (max_bound.x - min_bound.x) / 2.0;
        center.y = min_bound.y + (max_bound.y - min_bound.y) / 2.0;
        center.z = min_bound.z + (max_bound.z - min_bound.z) / 2.0;

        for (auto& v : vertices) {
            v.x -= center.x;
            v.y -= center.y;
            v.z -= center.z;
        }
        std::cout << ">>>> model centered.\n";
    }

    // --- 2. Calculate vertex normals ---
    std::cout << "\n>>>> [3/5] calculating vertex normals (in parallel)...\n";
    std::vector<Vector3> normals(vertices.size(), { 0.0, 0.0, 0.0 });

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (int y = 0; y < height - downscaleFactor; y += downscaleFactor) {
            for (int x = 0; x < width - downscaleFactor; x += downscaleFactor) {
              
                size_t idx_tl = static_cast<size_t>(y) * width + x;
                size_t idx_tr = static_cast<size_t>(y) * width + (x + downscaleFactor);
                size_t idx_bl = (static_cast<size_t>(y) + downscaleFactor) * width + x;
                size_t idx_br = (static_cast<size_t>(y) + downscaleFactor) * width + (x + downscaleFactor);

                int v1_idx = vertexMap[idx_tl];
                int v2_idx = vertexMap[idx_tr];
                int v3_idx = vertexMap[idx_bl];
                int v4_idx = vertexMap[idx_br];

                if (v1_idx == 0 || v2_idx == 0 || v3_idx == 0 || v4_idx == 0) continue;

                const auto& p1 = vertices[v1_idx - 1];
                const auto& p2 = vertices[v2_idx - 1];
                const auto& p3 = vertices[v3_idx - 1];
                const auto& p4 = vertices[v4_idx - 1];

                Vector3 faceNormal1 = cross(p3 - p1, p4 - p1);

#pragma omp atomic
                normals[v1_idx - 1].x += faceNormal1.x;
#pragma omp atomic
                normals[v1_idx - 1].y += faceNormal1.y;
#pragma omp atomic
                normals[v1_idx - 1].z += faceNormal1.z;

#pragma omp atomic
                normals[v3_idx - 1].x += faceNormal1.x;
#pragma omp atomic
                normals[v3_idx - 1].y += faceNormal1.y;
#pragma omp atomic
                normals[v3_idx - 1].z += faceNormal1.z;

#pragma omp atomic
                normals[v4_idx - 1].x += faceNormal1.x;
#pragma omp atomic
                normals[v4_idx - 1].y += faceNormal1.y;
#pragma omp atomic
                normals[v4_idx - 1].z += faceNormal1.z;

                Vector3 faceNormal2 = cross(p4 - p1, p2 - p1);
#pragma omp atomic
                normals[v1_idx - 1].x += faceNormal2.x;
#pragma omp atomic
                normals[v1_idx - 1].y += faceNormal2.y;
#pragma omp atomic
                normals[v1_idx - 1].z += faceNormal2.z;

#pragma omp atomic
                normals[v4_idx - 1].x += faceNormal2.x;
#pragma omp atomic
                normals[v4_idx - 1].y += faceNormal2.y;
#pragma omp atomic
                normals[v4_idx - 1].z += faceNormal2.z;

#pragma omp atomic
                normals[v2_idx - 1].x += faceNormal2.x;
#pragma omp atomic
                normals[v2_idx - 1].y += faceNormal2.y;
#pragma omp atomic
                normals[v2_idx - 1].z += faceNormal2.z;
            }
        }

#pragma omp for
        for (int i = 0; i < normals.size(); ++i) {
            normals[i].normalize();
        }
    }
    std::cout << ">>>> normals calculated and normalized.\n";

    // --- 3. Write all data to OBJ file---
    std::cout << "\n>>>> [4/5] writing data to .obj file...\n";
    std::ofstream file(outputFilePath);
    if (!file.is_open()) {
        throw std::runtime_error(">>>> failed to open output file: " + outputFilePath);
    }
    file.imbue(std::locale::classic());
    file << std::fixed << std::setprecision(6);

    file << "# OBJ file generated by DTM-to-Mesh converter\n";
    file << "o DTM_Mesh\n"; 

    file << "# Vertices: " << vertices.size() << "\n";
    for (const auto& v : vertices) {
        file << "v " << v.x << " " << v.y << " " << v.z << "\n";
    }

    file << "\n# Vertex Normals: " << normals.size() << "\n";
    for (const auto& vn : normals) {
        file << "vn " << vn.x << " " << vn.y << " " << vn.z << "\n";
    }

    file << "\ns off\n"; // Smoothing off eklendi

    // --- 4. Write faces with normal indices ---
    std::cout << "\n>>>> [5/5] writing faces with normal data...\n";
    long long faceCount = 0;
    file << "\n# Faces\n";
    for (int y = 0; y < height - downscaleFactor; y += downscaleFactor) {
        for (int x = 0; x < width - downscaleFactor; x += downscaleFactor) {
            int v1 = vertexMap[static_cast<size_t>(y) * width + x];
            int v2 = vertexMap[static_cast<size_t>(y) * width + (x + downscaleFactor)];
            int v3 = vertexMap[(static_cast<size_t>(y) + downscaleFactor) * width + x];
            int v4 = vertexMap[(static_cast<size_t>(y) + downscaleFactor) * width + (x + downscaleFactor)];

            if (v1 == 0 || v2 == 0 || v3 == 0 || v4 == 0) continue;

            file << "f " << v1 << "//" << v1 << " " << v3 << "//" << v3 << " " << v4 << "//" << v4 << "\n";
            file << "f " << v1 << "//" << v1 << " " << v4 << "//" << v4 << " " << v2 << "//" << v2 << "\n";
            faceCount += 2;
        }
    }
    file << "\n# Total faces: " << faceCount << "\n";
    std::cout << ">>>> written " << faceCount << " faces.\n";

    auto total_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start);
    std::cout << "\n>>>> successfully generated mesh: " << outputFilePath << " in " << duration.count() << "s.\n";
}

void MeshGenerator::generateHeightmap(const DigitalTerrainModel& dtm, const std::string& outputFilePath) const
{
	std::cout << ">>>> generating heightmap..." << std::endl;

	const int width = dtm.getWidth();
	const int height = dtm.getHeight();
	const double noDataValue = dtm.getNoDataValue();

	// read all raster data into memory, line by line ---
	std::vector<float> all_data;
	all_data.reserve(static_cast<size_t>(width) * height);
	std::vector<float> scanlineBuffer;

	for (int y = 0; y < height; ++y)
	{
		if (!dtm.readScanline(y, scanlineBuffer))
		{
			std::cerr << ">>>> warning: failed to read scanline " << y << std::endl;
		}
		all_data.insert(all_data.end(), scanlineBuffer.begin(), scanlineBuffer.end());
	}

	if (all_data.size() != static_cast<size_t>(width) * height)
	{
		throw std::runtime_error(">>>> failed to read all raster data correctly.");
	}

	// find the minimum and maximum valid elevation values ---
	double minElevation = std::numeric_limits<double>::max();
	double maxElevation = std::numeric_limits<double>::lowest();

	for (const float& val : all_data)
	{
		if (val != noDataValue)
		{
			if (val < minElevation) minElevation = val;
			if (val > maxElevation) maxElevation = val;
		}
	}

	if (minElevation >= maxElevation)
	{
		throw std::runtime_error(">>>> could not determine valid elevation range.");
	}
	std::cout << ">>>> elevation range: " << minElevation << "m to " << maxElevation << "m." << std::endl;

	const double elevationRange = maxElevation - minElevation;

	// normalize data and convert to 8-bit grayscale values
	std::vector<unsigned char> image_buffer(width * height); 

	for (size_t i = 0; i < all_data.size(); ++i)
	{
		double current_elevation = all_data[i];
		if (current_elevation == noDataValue)
		{
			image_buffer[i] = 0; // Black for NoData areas
		}
		else
		{
			// normalize the elevation to a 0.0 - 1.0 range
			//double normalized = (current_elevation - minElevation) / elevationRange;
			double normalized = (maxElevation - current_elevation) / elevationRange;

			// Scale to the 8-bit range (0 - 255) and cast to unsigned char.
			image_buffer[i] = static_cast<unsigned char>(std::round(normalized * 255.0)); 
		}
	}

	// save the data as an 8-bit grayscale PNG file ---
	const int components = 1; // 1 for grayscale

	// the stride must now reflect the size of an unsigned char (1 byte).
	const int stride_in_bytes = width * sizeof(unsigned char); 

	if (stbi_write_png(outputFilePath.c_str(), width, height, components, image_buffer.data(), stride_in_bytes))
	{
		std::cout << ">>>> successfully generated 8-bit heightmap: " << outputFilePath << std::endl; 
	}
	else
	{
		throw std::runtime_error(">>>> failed to write PNG heightmap file.");
	}
}