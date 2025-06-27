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

void MeshGenerator::writeVertices(std::ofstream& file, const DigitalTerrainModel& dtm, double zScale, std::vector<int>& vertexMap, int downscaleFactor) const
{
	std::cout << ">>>> writing vertices" << std::endl;

	const double* geoTransform = dtm.getGeoTransform();
	const float noDataValue = dtm.getNoDataValue();
	int width = dtm.getWidth();
	int height = dtm.getHeight();

	std::vector<float> scanlineBuffer(width);

	int currentVertexIndex = 1;

	for (int y = 0; y < height; y += downscaleFactor)
	{
		if (!dtm.readScanline(y, scanlineBuffer))
		{
			std::cerr << ">>>> failed to read scanline " << y << std::endl;
			continue;
		}

		for (int x = 0; x < width; x += downscaleFactor)
		{
			float elevation = scanlineBuffer[x];
			if (elevation == noDataValue)
				continue; // Skip NoData values

			// Calculate the geographic coordinates
			double worldX = geoTransform[0] + x * geoTransform[1] + y * geoTransform[2];
			double worldY = geoTransform[3] + x * geoTransform[4] + y * geoTransform[5];
			double worldZ = -elevation * zScale;

            file << "v " << worldX << " " << worldY << " " << worldZ << "\n";
			vertexMap[y * width + x] = currentVertexIndex++;
		}

		if ((y + 1) % 100000 == 0)
		{
			std::cout << ">>>> processed " << (y + 1) << " of " << height << " lines for vertices.\n";
		}
	}

	file << "# Total vertices: " << (currentVertexIndex - 1) << "\n\n";
	std::cout << ">>>> # total vertices: " << (currentVertexIndex - 1) << "\n\n";
}

void MeshGenerator::writeFaces(std::ofstream& file, const DigitalTerrainModel& dtm, const std::vector<int>& vertexMap, int downscaleFactor) const
{
	std::cout << ">>>> writing faces" << std::endl;

	const int width = dtm.getWidth();
	const int height = dtm.getHeight();
	long long faceCount = 0; // to keep millions of faces

	// a vector of stringstreams, each thread could write to its own stream
	// this avoids race condition on the single output file stream
	std::vector<std::stringstream> privateBuffers;

	#pragma omp parallel
	{
		// Each thread gets its own private stringstream buffer.
		// This is crucial to prevent multiple threads from writing to the same memory at once.
		std::stringstream localBuffer;

		// Distribute the 'y' loop iterations among the available threads.
		// 'schedule(static)' divides the work into contiguous (adjacent) chunks and gives one to each thread.
		// 'reduction(+:totalFaceCount)' creates a private copy of totalFaceCount for each thread,
		// and safely combines them (sums them up) at the end of the parallel region.
		#pragma omp for schedule(static) reduction(+:faceCount)
		for (int y = 0; y < height - downscaleFactor; y += downscaleFactor)
		{
			for (int x = 0; x < width - downscaleFactor; x += downscaleFactor)
			{
				// Get the four corner vertices of the larger quad, spaced by 'downscaleFactor'.
				int v1 = vertexMap[static_cast<size_t>(y) * width + x];                  // Top-left
				int v2 = vertexMap[static_cast<size_t>(y) * width + (x + downscaleFactor)];          // Top-right
				int v3 = vertexMap[(static_cast<size_t>(y) + downscaleFactor) * width + x];          // Bottom-left
				int v4 = vertexMap[(static_cast<size_t>(y) + downscaleFactor) * width + (x + downscaleFactor)]; // Bottom-right

				if (v1 == 0 || v2 == 0 || v3 == 0 || v4 == 0)
					continue;

				// Write the output to the thread's private local buffer, not the main file stream.
				// two triangles for the quad formed by the vertices
				localBuffer << "f " << v1 << " " << v3 << " " << v4 << "\n";
				localBuffer << "f " << v1 << " " << v4 << " " << v2 << "\n";
				faceCount += 2;
			}
		}

		// A critical section ensures that only one thread at a time can execute this block.
		// This is necessary to safely push the thread's local buffer into the shared 'privateBuffers' vector.
		#pragma omp critical
		{
			// std::move is used for efficiency, avoiding a potentially expensive copy of the stringstream.
			privateBuffers.push_back(std::move(localBuffer));
		}
	}

	// At this point, all parallel processing is complete.
	// Now, write the contents of each thread's buffer to the main file stream sequentially.
	// This ensures the output in the final file remains in the correct order.
	std::cout << ">>>> flushing parallel buffers to file" << std::endl;
	for (const auto& buffer : privateBuffers)
	{
		file << buffer.rdbuf();
	}

	file << "# Total faces: " << faceCount << "\n";
	std::cout << ">>>> # total faces: " << faceCount << "\n";
}

void MeshGenerator::generateMesh(const DigitalTerrainModel& dtm, const std::string& outputFilePath, int downscaleFactor, double zScale) const
{
	auto total_start = std::chrono::high_resolution_clock::now();

	std::cout << ">>>> generating mesh for DTM" << std::endl;
	std::ofstream file(outputFilePath);

	// for number formatting, which always uses a period '.' as the decimal separator,
    // ensuring the .obj file is universally compatible.
    file.imbue(std::locale::classic());

	if (!file.is_open())
	{
		throw std::runtime_error(">>>> failed to open output file: " + outputFilePath);
	}

	// map to store vertex indices, 1-based indexing
	// a value of 0 indicates NoData
	std::vector<int> vertexMap(dtm.getWidth() * dtm.getHeight(), 0);

	//set precision for floating point numbers, i.e do not use scientific notation
	file << std::fixed;

	file << "# OBJ file generated by LROC DTM to Mesh Converter\n";
	file << "# Source DTM: " << dtm.getWidth() << "x" << dtm.getHeight() << " pixels\n";
	file << "# Vertical exaggeration: " << zScale << "\n\n";

	// write vertices
	std::cout << "\n>>>> [1/2] writing vertices...\n";
	auto vertices_start = std::chrono::high_resolution_clock::now();
	writeVertices(file, dtm, zScale, vertexMap, downscaleFactor);
	auto vertices_end = std::chrono::high_resolution_clock::now();
	// Calculate the duration and cast it to milliseconds for readability.
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(vertices_end - vertices_start);
	std::cout << ">>>> writeVertices completed in " << duration.count() << " ms.\n";

	// write faces
	std::cout << "\n>>>> [2/2] writing faces...\n";
	auto faces_start = std::chrono::high_resolution_clock::now();
	writeFaces(file, dtm, vertexMap, downscaleFactor);
	auto faces_end = std::chrono::high_resolution_clock::now();
	// Calculate the duration and cast it to milliseconds.
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(faces_end - faces_start);
	std::cout << ">>>> writeFaces completed in " << duration.count() << " ms.\n";

	std::cout << ">>>> successfully generated mesh: " << outputFilePath << std::endl;
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