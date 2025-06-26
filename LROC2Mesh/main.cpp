#include "gdal_priv.h"
#include <iostream>
#include <vector>

int main() {
    GDALAllRegister();
    const char* filename = "NAC_DTM_APOLLO11.TIF";

    GDALDataset* dataset = (GDALDataset*)GDALOpen(filename, GA_ReadOnly);
    if (!dataset) {
        std::cerr << "Failed to open file.\n";
        return 1;
    }

    GDALRasterBand* band = dataset->GetRasterBand(1);
    int width = band->GetXSize();
    int height = band->GetYSize();

    std::vector<float> data(width * height);
    band->RasterIO(GF_Read, 0, 0, width, height, data.data(), width, height, GDT_Float32, 0, 0);

    std::cout << "Read elevation data: " << width << "x" << height << "\n";

    GDALClose(dataset);
    return 0;
}