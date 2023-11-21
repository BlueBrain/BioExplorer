/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "DICOMLoader.h"

#include <plugin/common/Logs.h>

#include <platform/core/common/utils/Utils.h>
#include <platform/core/engineapi/Model.h>
#include <platform/core/engineapi/Scene.h>
#include <platform/core/engineapi/SharedDataVolume.h>

#include <dcmtk/dcmdata/dcddirif.h>
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmimgle/dcmimage.h>

#include <boost/filesystem.hpp>

namespace medicalimagingexplorer
{
namespace dicom
{
using namespace core;

const std::string SUPPORTED_BASENAME_DICOMDIR = "DICOMDIR";
const std::string SUPPORTED_EXTENSION_DCM = "dcm";
const ColorMap colormap = {"DICOM",
                           {{0.21960784494876862, 0.0, 0.0},
                            {0.43921568989753723, 0.0, 0.0},
                            {0.6666666865348816, 0.16470588743686676, 0.0},
                            {0.886274516582489, 0.3843137323856354, 0.0},
                            {1.0, 0.6117647290229797, 0.11372549086809158},
                            {1.0, 0.8313725590705872, 0.3294117748737335},
                            {1.0, 1.0, 0.5607843399047852},
                            {1.0, 1.0, 0.7764706015586853}}};
const Vector2ds controlPoints = {{0.0, 0.0},   {0.125, 0.0}, {0.25, 0.0},  {0.375, 0.0}, {0.5, 1.0},
                                 {0.625, 1.0}, {0.75, 1.0},  {0.875, 1.0}, {1.0, 1.0}};

std::string getDataTypeAsString(const DataType dataType)
{
    switch (dataType)
    {
    case DataType::INT8:
        return "INT8";
    case DataType::UINT8:
        return "UINT8";
    case DataType::INT16:
        return "INT16";
    case DataType::UINT16:
        return "UINT16";
    case DataType::FLOAT:
        return "FLOAT";
    default:
        return "UNKNOWN";
    }
}

DICOMLoader::DICOMLoader(Scene& scene, const GeometryParameters& geometryParameters, PropertyMap&& loaderParams)
    : Loader(scene)
    , _geometryParameters(geometryParameters)
    , _loaderParams(loaderParams)
{
}

void DICOMLoader::_setDefaultTransferFunction(Model& model, const Vector2f& dataRange) const
{
    auto& tf = model.getTransferFunction();
    tf.setValuesRange(dataRange);
    tf.setColorMap(colormap);
    tf.setControlPoints(controlPoints);
}

void DICOMLoader::_readDICOMFile(const std::string& fileName, DICOMImageDescriptor& imageDescriptor) const
{
    DcmFileFormat file;
    file.loadFile(fileName.c_str());
    DcmDataset* dataset = file.getDataset();
    double position[3];
    for (size_t i = 0; i < 3; ++i)
        dataset->findAndGetFloat64(DCM_ImagePositionPatient, position[i], i);
    long imageSize[2];
    dataset->findAndGetLongInt(DCM_Columns, imageSize[0], 0);
    dataset->findAndGetLongInt(DCM_Rows, imageSize[1], 0);
    double pixelSpacing[2];
    for (size_t i = 0; i < 2; ++i)
        dataset->findAndGetFloat64(DCM_PixelSpacing, pixelSpacing[i], i);

    DicomImage* image = new DicomImage(fileName.c_str());
    if (image)
    {
        imageDescriptor.nbFrames = image->getNumberOfFrames();
        PLUGIN_DEBUG(fileName << ": " << imageDescriptor.nbFrames << " frame(s)");

        if (image->getStatus() != EIS_Normal)
            throw std::runtime_error("Error: cannot load DICOM image from " + fileName);
        const auto imageDepth = image->getDepth();
        size_t voxelSize;
        if (image->getInterData())
        {
            switch (image->getInterData()->getRepresentation())
            {
            case EPR_Sint8:
                imageDescriptor.dataType = DataType::INT8;
                voxelSize = sizeof(int8_t);
                break;
            case EPR_Uint8:
                imageDescriptor.dataType = DataType::UINT8;
                voxelSize = sizeof(uint8_t);
                break;
            case EPR_Sint16:
                imageDescriptor.dataType = DataType::INT16;
                voxelSize = sizeof(int16_t);
                break;
            case EPR_Uint16:
                imageDescriptor.dataType = DataType::UINT16;
                voxelSize = sizeof(uint16_t);
                break;
            default:
                throw std::runtime_error("Unsupported volume format");
            }
        }
        else
            throw std::runtime_error("Failed to identify image representation");

        imageDescriptor.position = {(float)position[0], (float)position[1], (float)position[2]};
        imageDescriptor.dimensions = {(unsigned int)imageSize[0], (unsigned int)imageSize[1]};
        imageDescriptor.pixelSpacing = {(float)pixelSpacing[0], (float)pixelSpacing[1]};

        const auto bufferSize = image->getOutputDataSize();
        imageDescriptor.buffer.resize(bufferSize);
        if (!image->getOutputData(imageDescriptor.buffer.data(), bufferSize, imageDepth))
            throw std::runtime_error("Failed to load image data from " + fileName);

        double minRange, maxRange;
        image->getMinMaxValues(minRange, maxRange);
        imageDescriptor.dataRange = {float(minRange), float(maxRange)};
        delete image;
    }
    else
        throw std::runtime_error("Failed to open " + fileName);
}

DICOMImageDescriptors DICOMLoader::_parseDICOMImagesData(const std::string& fileName, ModelMetadata& metadata) const
{
    DICOMImageDescriptors dicomImages;
    DcmDicomDir dicomdir(fileName.c_str());
    DcmDirectoryRecord* studyRecord = nullptr;
    DcmDirectoryRecord* patientRecord = nullptr;
    DcmDirectoryRecord* seriesRecord = nullptr;
    DcmDirectoryRecord* imageRecord = nullptr;
    OFString tmpString;

    if (!dicomdir.verify().good())
        throw std::runtime_error("Failed to open " + fileName);

    auto root = dicomdir.getRootRecord();
    while ((patientRecord = root.nextSub(patientRecord)) != nullptr)
    {
        patientRecord->findAndGetOFString(DCM_PatientID, tmpString);
        metadata["Patient ID"] = tmpString.c_str();
        patientRecord->findAndGetOFString(DCM_PatientName, tmpString);
        metadata["Patient name"] = tmpString.c_str();
        patientRecord->findAndGetOFString(DCM_PatientAge, tmpString);
        metadata["Patient age"] = tmpString.c_str();
        patientRecord->findAndGetOFString(DCM_PatientSex, tmpString);
        metadata["Patient sex"] = tmpString.c_str();
        patientRecord->findAndGetOFString(DCM_PatientBirthDate, tmpString);
        metadata["Patient date of birth"] = tmpString.c_str();

        while ((studyRecord = patientRecord->nextSub(studyRecord)) != nullptr)
        {
            studyRecord->findAndGetOFString(DCM_StudyID, tmpString);
            metadata["Study ID"] = tmpString.c_str();

            // Read all series and filter according to SeriesInstanceUID
            while ((seriesRecord = studyRecord->nextSub(seriesRecord)) != nullptr)
            {
                seriesRecord->findAndGetOFString(DCM_SeriesNumber, tmpString);
                PLUGIN_INFO("Series number: " << tmpString);

                size_t nbImages = 0;
                while ((imageRecord = seriesRecord->nextSub(imageRecord)) != nullptr)
                {
                    OFString refId;
                    imageRecord->findAndGetOFStringArray(DCM_ReferencedFileID, refId);

                    // Replace backslashes with slashes
                    std::string str = std::string(refId.data());
                    while (str.find("\\") != std::string::npos)
                        str.replace(str.find("\\"), 1, "/");

                    // Full image filename
                    boost::filesystem::path path = fileName;
                    boost::filesystem::path folder = path.parent_path();
                    const std::string imageFileName = std::string(folder.string()) + "/" + str;

                    // Load image from file
                    DICOMImageDescriptor imageDescriptor;
                    _readDICOMFile(imageFileName, imageDescriptor);
                    dicomImages.push_back(imageDescriptor);
                    ++nbImages;
                }
                PLUGIN_DEBUG(nbImages << " images");
                // break; // TODO: Manage multiple series
            }
        }
    }
    return dicomImages;
}

ModelDescriptorPtr DICOMLoader::_readFile(const std::string& fileName) const
{
    DICOMImageDescriptor imageDescriptor;
    _readDICOMFile(fileName, imageDescriptor);

    // Data range
    const Vector2f dataRange = {std::numeric_limits<uint16_t>::max(), std::numeric_limits<uint16_t>::min()};

    // Create Model
    auto model = _scene.createModel();
    if (!model)
        throw std::runtime_error("Failed to create model");

    auto volume = model->createSharedDataVolume({imageDescriptor.dimensions.x, imageDescriptor.dimensions.y, 1},
                                                {imageDescriptor.pixelSpacing.x, imageDescriptor.pixelSpacing.y, 1},
                                                DataType::UINT16);
    if (!volume)
        throw std::runtime_error("Failed to create volume");

    volume->setDataRange(dataRange);
    volume->mapData(imageDescriptor.buffer);
    model->addVolume(VOLUME_MATERIAL_ID, volume);

    // Transfer function initialization
    _setDefaultTransferFunction(*model, dataRange);

    // Transformation
    Transformation transformation;
    transformation.setRotationCenter(model->getBounds().getCenter());
    ModelMetadata metaData = {{"Dimensions", to_string(imageDescriptor.dimensions)},
                              {"Element spacing", to_string(imageDescriptor.pixelSpacing)}};

    auto modelDescriptor = std::make_shared<ModelDescriptor>(std::move(model), fileName, metaData);
    modelDescriptor->setTransformation(transformation);
    return modelDescriptor;
}

ModelDescriptorPtr DICOMLoader::_readDirectory(const std::string& fileName, const LoaderProgress& callback) const
{
    ModelMetadata metaData;
    const auto& dicomImages = _parseDICOMImagesData(fileName, metaData);

    if (dicomImages.empty())
        throw std::runtime_error("DICOM folder does not contain any images");

    // Dimensions
    Vector3ui dimensions = {dicomImages[0].dimensions.x, dicomImages[0].dimensions.y, (unsigned int)dicomImages.size()};

    // Element spacing (if single image, assume that z pixel spacing is the
    // same as y
    Vector3f spacing{dicomImages[0].pixelSpacing.x, dicomImages[0].pixelSpacing.y, dicomImages[0].pixelSpacing.y};
    if (dicomImages.size() > 1)
        spacing.z = dicomImages[1].position.z - dicomImages[0].position.z;

    // Load images into volume
    callback.updateProgress("Loading voxels ...", 0.5f);

    // Data type and range
    DataType dataType;
    Vector2f dataRange{std::numeric_limits<float>::max(), std::numeric_limits<float>::min()};

    // Compile images into one single volume
    uint8_ts volumeData;
    for (const auto& dicomImage : dicomImages)
    {
        volumeData.insert(volumeData.end(), dicomImage.buffer.begin(), dicomImage.buffer.end());
        dataType = dicomImage.dataType;
        dataRange.x = std::min(dataRange.x, dicomImage.dataRange.x);
        dataRange.y = std::max(dataRange.y, dicomImage.dataRange.y);
    }

    // Create Model
    callback.updateProgress("Creating model ...", 1.f);
    PLUGIN_INFO("Creating " << _dataTypeToString(dataType) << " volume " << dimensions << ", " << spacing << ", "
                            << dataRange << " (" << volumeData.size() << " bytes)");
    auto model = _scene.createModel();
    if (!model)
        throw std::runtime_error("Failed to create model");

    auto volume = model->createSharedDataVolume(dimensions, spacing, dataType);
    if (!volume)
        throw std::runtime_error("Failed to create volume");

    volume->setDataRange(dataRange);
    volume->mapData(volumeData);
    model->addVolume(VOLUME_MATERIAL_ID, volume);

    // Transfer function initialization
    _setDefaultTransferFunction(*model, dataRange);

    // Transformation
    Transformation transformation;
    transformation.setRotationCenter(model->getBounds().getCenter());
    metaData["Dimensions"] =
        std::to_string(dimensions.x) + "," + std::to_string(dimensions.y) + "," + std::to_string(dimensions.z);
    metaData["Spacing"] = std::to_string(spacing.x) + "," + std::to_string(spacing.y) + "," + std::to_string(spacing.z);
    metaData["Data range"] = std::to_string(dataRange.x) + "," + std::to_string(dataRange.y);
    metaData["Data type"] = getDataTypeAsString(dataType);

    auto modelDescriptor = std::make_shared<ModelDescriptor>(std::move(model), "DICOMDIR", metaData);
    modelDescriptor->setTransformation(transformation);
    return modelDescriptor;
}

ModelDescriptorPtr DICOMLoader::importFromFolder(const std::string& path)
{
    auto files = parseFolder(path, {"." + SUPPORTED_EXTENSION_DCM});
    std::sort(files.begin(), files.end());
    if (files.empty())
        throw std::runtime_error("DICOM folder does not contain any images");

    DICOMImageDescriptors imageDescriptors;
    imageDescriptors.resize(files.size());

    Vector3ui dimensions;
    Vector3f spacing = {1, 1, 1};

    Vector2f dataRange{std::numeric_limits<float>::max(), std::numeric_limits<float>::min()};
    DataType dataType;

    // Load remaining images
    size_t i = 0;
    for (const auto& file : files)
    {
        auto& id = imageDescriptors[i];
        _readDICOMFile(file, id);

        switch (i)
        {
        case 0:
        {
            dataType = id.dataType;
            dimensions = {id.dimensions.x, id.dimensions.y, id.nbFrames};
            spacing = {id.pixelSpacing.x, id.pixelSpacing.y, 1.f};
            break;
        }
        case 1:
            spacing.z = abs(imageDescriptors[i].position.z - imageDescriptors[i - 1].position.z);
            break;
        default:
            dimensions.z += id.nbFrames;
        }

        dataRange.x = std::min(dataRange.x, id.dataRange.x);
        dataRange.y = std::max(dataRange.y, id.dataRange.y);

        ++i;
    }

    // Create volume
    PLUGIN_INFO("Creating " << _dataTypeToString(dataType) << " volume " << dimensions << ", " << spacing << ", "
                            << dataRange);

    uint8_ts volumeData;
    for (const auto& id : imageDescriptors)
        volumeData.insert(volumeData.end(), id.buffer.begin(), id.buffer.end());

    // Create Model
    auto model = _scene.createModel();
    if (!model)
        throw std::runtime_error("Failed to create model");

    auto volume = model->createSharedDataVolume(dimensions, spacing, dataType);
    if (!volume)
        throw std::runtime_error("Failed to create volume");

    volume->setDataRange(dataRange);
    volume->mapData(volumeData);
    model->addVolume(VOLUME_MATERIAL_ID, volume);

    Transformation transformation;
    transformation.setRotationCenter(model->getBounds().getCenter());
    ModelMetadata metaData = {{"Dimensions", to_string(dimensions)},
                              {"Element spacing", to_string(spacing)},
                              {"Data range", to_string(dataRange)}};
    auto modelDescriptor = std::make_shared<ModelDescriptor>(std::move(model), path, metaData);
    modelDescriptor->setTransformation(transformation);
    modelDescriptor->setBoundingBox(true);
    return modelDescriptor;
}

ModelDescriptorPtr DICOMLoader::importFromStorage(const std::string& storage, const LoaderProgress& callback,
                                                  const PropertyMap& /*properties*/) const
{
    PLUGIN_INFO("Importing DICOM dataset from " << storage);
    const auto extension = boost::filesystem::extension(storage);
    if (extension == "." + SUPPORTED_EXTENSION_DCM)
        return _readFile(storage);
    return _readDirectory(storage, callback);
}

ModelDescriptorPtr DICOMLoader::importFromBlob(Blob&&, const LoaderProgress&, const PropertyMap&) const
{
    throw std::runtime_error("Loading DICOM from blob is not supported");
}

std::string DICOMLoader::getName() const
{
    return "Loader for DICOM datasets";
}

std::vector<std::string> DICOMLoader::getSupportedStorage() const
{
    return {SUPPORTED_EXTENSION_DCM, SUPPORTED_BASENAME_DICOMDIR};
}

bool DICOMLoader::isSupported(const std::string& storage, const std::string& extension) const
{
    const auto basename = boost::filesystem::basename(storage);
    const std::set<std::string> basenames = {SUPPORTED_BASENAME_DICOMDIR};
    const std::set<std::string> extensions = {SUPPORTED_EXTENSION_DCM};
    return (basenames.find(basename) != basenames.end() || extensions.find(extension) != extensions.end());
}

std::string DICOMLoader::_dataTypeToString(const DataType& dataType) const
{
    switch (dataType)
    {
    case DataType::UINT8:
        return "Unsigned 8bit";
    case DataType::UINT16:
        return "Unsigned 16bit ";
    case DataType::UINT32:
        return "Unsigned 32bit";
    case DataType::INT8:
        return "Signed 8bit";
    case DataType::INT16:
        return "Signed 16bit";
    case DataType::INT32:
        return "Signed 32bit";
    case DataType::FLOAT:
        return "Float";
    case DataType::DOUBLE:
        return "Double";
    }
    return "Undefined";
}

PropertyMap DICOMLoader::getCLIProperties()
{
    PropertyMap pm("DICOMLoader");
    return pm;
}
} // namespace dicom
} // namespace medicalimagingexplorer
