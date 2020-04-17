#ifndef COVID19_PROTEIN_H
#define COVID19_PROTEIN_H

#include <brayns/engineapi/Model.h>
#include <common/types.h>

/**
 * @brief The Protein class
 */
class Protein
{
public:
    Protein(brayns::Scene& scene, const std::string& name,
            const std::string& filename, const float radiusMultiplier = 1.f);

    // Color schemes
    void setColorScheme(const ColorScheme& colorScheme, const Palette& palette);

    // Amino acid sequence
    StringMap getSequencesAsString() const;
    void setAminoAcidSequence(const std::string& aminoAcidSequence)
    {
        _aminoAcidSequence = aminoAcidSequence;
    }
    const std::string& getAminoAcidSequence() const
    {
        return _aminoAcidSequence;
    }

    // Class member accessors
    AtomMap& getAtoms() { return _atomMap; }
    Residues& getResidues() { return _residues; }
    SequenceMap& getSequences() { return _sequenceMap; }
    brayns::ModelDescriptorPtr getModelDescriptor() { return _modelDescriptor; }

private:
    // Color schemes
    void _setAtomColorScheme();
    void _setChainColorScheme(const Palette& palette);
    void _setResiduesColorScheme(const Palette& palette);
    void _setAminoAcidSequenceColorScheme(const Palette& palette);
    void _setGlycosylationSiteColorScheme(const Palette& palette);

    void _setMaterialDiffuseColor(const size_t atomIndex,
                                  const RGBColor& color);
    void _setMaterialDiffuseColor(const size_t atomIndex, const Color& color);

    // IO
    void _readAtom(const std::string& line);
    void _readSequence(const std::string& line);
    void _readTitle(const std::string& line);
    void _readRemark(const std::string& line);
    void _readConnect(const std::string& line);

    // Class members
    AtomMap _atomMap;
    Residues _residues;
    SequenceMap _sequenceMap;
    BondsMap _bondsMap;

    std::string _aminoAcidSequence;
    std::string _title;
    brayns::ModelDescriptorPtr _modelDescriptor{nullptr};
};

#endif // COVID19_PROTEIN_H
