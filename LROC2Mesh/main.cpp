#pragma once
#include <iostream>
#include <vector>

#include "MeshGenerator.h"

int main(int argc, char* argv[])
{
    std::string inputFile;
    std::string outputFile;
    double zScale;

	// no arguments provided, use default values
    if (argc == 1)
    {
        std::cout << ">>>> running with default debug values.\n";
        inputFile = "NAC_DTM_APOLLO11.TIF"; 
        outputFile = "apollo11.obj";   
        zScale = 1.0;                      
    }
    else if (argc == 3 || argc == 4)
    {
        std::cout << ">>>> running with command-line arguments.\n";
        inputFile = argv[1];
        outputFile = argv[2];
		// If a third argument is provided, use it as the zScale, otherwise default to 1.0
        zScale = (argc == 4) ? std::stod(argv[3]) : 1.0;
    }
    else
    {
        std::cerr << ">>>> usage: " << argv[0] << " <input.tif> <output.obj> [z_scale]\n";
        std::cerr << ">>>>  z_scale (optional): A factor to exaggerate the vertical terrain. Default is 1.0.\n";
        return 1; 
    }


    // Initialise GDAL
    GDALAllRegister();

    try
    {
        std::cout << ">>>> input file: " << inputFile << std::endl;
        std::cout << ">>>> output file: " << outputFile << std::endl;
        std::cout << ">>>> vertical exaggeration (zScale): " << zScale << std::endl;

        DigitalTerrainModel dtm(inputFile);
        MeshGenerator generator;
        generator.generateMesh(dtm, outputFile, zScale);
    }
    catch (const std::exception& e)
    {
        std::cerr << "\n>>>> an error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}