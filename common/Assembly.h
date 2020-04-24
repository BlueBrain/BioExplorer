#ifndef ASSEMBLY_H
#define ASSEMBLY_H

#include <common/types.h>

class Assembly
{
public:
    Assembly(brayns::Scene &scene,
             const AssemblyDescriptor &assemblyDescriptor);
    ~Assembly();

    void setColorScheme(const ColorSchemeDescriptor &csd);
    void setAminoAcidSequence(const AminoAcidSequenceDescriptor &aasd);
    std::string getAminoAcidSequences(
        const AminoAcidSequencesDescriptor &payload) const;
    void addRNASequence(const RNASequenceDescriptor &rd);
    void addProtein(const ProteinDescriptor &pd);
    void addMesh(const MeshDescriptor &md);

private:
    void _processInstances(brayns::ModelDescriptorPtr md,
                           const float assemblyRadius, const size_t occurrences,
                           const size_t randomSeed,
                           const brayns::Quaterniond &orientation,
                           const ModelContentType &modelType,
                           const float locationCutoffAngle = 0.f);

    brayns::Vector3f _position;
    bool _halfStructure;
    brayns::Scene &_scene;
    ProteinMap _proteins;
    MeshMap _meshes;
    std::vector<std::pair<brayns::Vector3f, float>> _occupiedDirections;
};

#endif // ASSEMBLY_H
