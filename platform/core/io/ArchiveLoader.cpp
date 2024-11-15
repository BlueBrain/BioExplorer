/*
    Copyright 2018 - 2024 Blue Brain Project / EPFL

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include "ArchiveLoader.h"

#include <platform/core/common/Logs.h>
#include <platform/core/common/utils/FileSystem.h>
#include <platform/core/common/utils/Utils.h>
#include <platform/core/engineapi/Model.h>
#include <platform/core/engineapi/Scene.h>

#include <fstream>

#include <archive.h>
#include <archive_entry.h>

namespace
{
bool isSupportedArchiveType(const std::string& extension)
{
    // No way to get all supported types from libarchive...
    // Extend this list if you feel your favorite archive type should be here
    const std::set<std::string> extensions{"zip", "gz", "tgz", "bz2", "rar"};
    return extensions.find(extension) != extensions.end();
}

archive* _openArchive(const std::string& filename)
{
    auto archive = archive_read_new();
    archive_read_support_format_all(archive);
    archive_read_support_filter_all(archive);

    // non-tar archives like gz, bzip2, ... need to be added as raw
    auto extension = core::extractExtension(filename);
    if (!extension.empty())
    {
        if (isSupportedArchiveType(extension))
            archive_read_support_format_raw(archive);
    }
    if (archive_read_open_filename(archive, filename.c_str(), 10240) == ARCHIVE_OK)
    {
        return archive;
    }
    archive_read_free(archive);
    return nullptr;
}

archive* _openArchive(const core::Blob& blob)
{
    auto archive = archive_read_new();
    archive_read_support_format_all(archive);
    archive_read_support_filter_all(archive);

    // non-tar archives like gz, bzip2, ... need to be added as raw
    auto extension = core::extractExtension(blob.name);
    if (!extension.empty())
    {
        if (isSupportedArchiveType(extension))
            archive_read_support_format_raw(archive);
    }
    if (archive_read_open_memory(archive, (void*)blob.data.data(), blob.data.size()) == ARCHIVE_OK)
    {
        return archive;
    }
    archive_read_free(archive);
    return nullptr;
}

int copy_data(struct archive* ar, struct archive* aw)
{
    for (;;)
    {
        const void* buff;
        size_t size;
        int64_t offset;

        auto r = archive_read_data_block(ar, &buff, &size, &offset);
        if (r == ARCHIVE_EOF)
            return ARCHIVE_OK;
        if (r < ARCHIVE_OK)
            return r;
        r = archive_write_data_block(aw, buff, size, offset);
        if (r < ARCHIVE_OK)
        {
            std::cerr << archive_error_string(aw) << std::endl;
            return (r);
        }
    }
}

void _extractArchive(archive* archive, const std::string& filename, const std::string& destination)
{
    auto writer = archive_write_disk_new();
    archive_write_disk_set_options(writer, 0);
    archive_write_disk_set_standard_lookup(writer);

    for (;;)
    {
        archive_entry* entry;
        auto r = archive_read_next_header(archive, &entry);
        if (r == ARCHIVE_EOF)
            break;
        if (r < ARCHIVE_OK)
            std::cerr << archive_error_string(archive) << std::endl;
        if (r < ARCHIVE_WARN)
        {
            throw std::runtime_error(std::string("Error reading file from archive: ") + archive_error_string(archive));
        }
        const char* currentFile = archive_entry_pathname(entry);

        // magic 'data' file for gzip archives is useless to us, so rename it
        if (std::string(currentFile) == "data")
            currentFile = filename.c_str();
        const std::string fullOutputPath = destination + "/" + currentFile;
        archive_entry_set_pathname(entry, fullOutputPath.c_str());
        r = archive_write_header(writer, entry);
        if (r < ARCHIVE_OK)
            std::cerr << archive_error_string(writer) << std::endl;
        else if (archive_entry_size(entry) > 0)
        {
            r = copy_data(archive, writer);
            if (r < ARCHIVE_OK)
                std::cerr << archive_error_string(writer) << std::endl;
            if (r < ARCHIVE_WARN)
                throw std::runtime_error(std::string("Error writing file from archive to disk: ") +
                                         archive_error_string(archive));
        }
        r = archive_write_finish_entry(writer);
        if (r < ARCHIVE_OK)
            std::cerr << archive_error_string(writer) << std::endl;
        if (r < ARCHIVE_WARN)
            throw std::runtime_error(std::string("Error finishing current file: ") + archive_error_string(archive));
    }
    archive_read_free(archive);
    archive_write_close(writer);
    archive_write_free(writer);
}

void extractFile(const std::string& filename, const std::string& destination)
{
    auto archive = _openArchive(filename);
    if (!archive)
        throw std::runtime_error(filename + " is not a supported archive type");
    _extractArchive(archive, fs::path(filename).stem().string(), destination);
}

void extractBlob(core::Blob&& blob, const std::string& destination)
{
    auto archive = _openArchive(blob);
    if (!archive)
        throw std::runtime_error("Blob is not a supported archive type");
    _extractArchive(archive, fs::path(blob.name).stem().string(), destination);
}

const auto LOADER_NAME = "archive";

struct TmpFolder
{
    TmpFolder()
    {
        if (!mkdtemp((char*)path.data()))
            throw std::runtime_error("Could not create temporary directory");
    }
    ~TmpFolder() { fs::remove_all(path); }
    std::string path{"/tmp/brayns_extracted_XXXXXX"};
};
} // namespace

namespace core
{
ArchiveLoader::ArchiveLoader(Scene& scene, LoaderRegistry& registry)
    : Loader(scene)
    , _registry(registry)
{
}

bool ArchiveLoader::isSupported(const std::string& storage, const std::string& extension) const
{
    return isSupportedArchiveType(extension);
}

ModelDescriptorPtr ArchiveLoader::loadExtracted(const std::string& path, const LoaderProgress& callback,
                                                const PropertyMap& properties) const
{
    const auto loaderName = properties.getProperty<std::string>("loaderName", "");
    const Loader* loader = loaderName.empty() ? nullptr : &_registry.getSuitableLoader("", "", loaderName);

    for (const auto& i : fs::directory_iterator(path))
    {
        const std::string currPath = i.path().string();
        const std::string extension = extractExtension(currPath);

        if (loader && loader->isSupported(currPath, extension))
        {
            return loader->importFromFile(currPath, callback, properties);
        }
        else if (!loader && _registry.isSupportedFile(currPath))
        {
            const auto& loaderMatch = _registry.getSuitableLoader(currPath, "", "");
            return loaderMatch.importFromFile(currPath, callback, properties);
        }
    }
    throw std::runtime_error("No loader found for archive.");
}

ModelDescriptorPtr ArchiveLoader::importFromBlob(Blob&& blob, const LoaderProgress& callback,
                                                 const PropertyMap& properties) const
{
    TmpFolder tmpFolder;
    extractBlob(std::move(blob), tmpFolder.path);
    return loadExtracted(tmpFolder.path, callback, properties);
}

ModelDescriptorPtr ArchiveLoader::importFromStorage(const std::string& storage, const LoaderProgress& callback,
                                                    const PropertyMap& properties) const
{
    TmpFolder tmpFolder;
    extractFile(filename, tmpFolder.path);
    return loadExtracted(tmpFolder.path, callback, properties);
}

std::string ArchiveLoader::getName() const
{
    return LOADER_NAME;
}

std::vector<std::string> ArchiveLoader::getSupportedStorage() const
{
    return {"zip", "gz", "tgz", "bz2", "rar"};
}
} // namespace core
