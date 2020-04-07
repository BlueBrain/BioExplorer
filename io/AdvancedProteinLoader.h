/* Copyright (c) 2020, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef ADVANCEDPROTEINLOADER_H
#define ADVANCEDPROTEINLOADER_H

#include <brayns/common/loader/Loader.h>
#include <brayns/parameters/GeometryParameters.h>

/** Define the color scheme to be applied to the geometry */
enum class ColorScheme
{
    none = 0,
    atoms = 1,
    chains = 2,
    residues = 3,
    location = 4
};

// Loader properties
const brayns::Property PROP_RADIUS_MULTIPLIER = {"radiusMultiplier",
                                                 1.,
                                                 {"Radius multiplier"}};
const brayns::Property PROP_PROTEIN_COLOR_SCHEME = {
    "colorScheme",
    brayns::enumToString(ColorScheme::none),
    brayns::enumNames<ColorScheme>(),
    {"Color scheme",
     "Color scheme to be applied to the proteins "
     "[none|atoms|chains|residues|location"}};

const brayns::Property PROP_TRANSMEMBRANE_SEQUENCE = {
    "transmembraneSequence",
    std::string(""),
    {"Transmembrane Sequence",
     "Sequence of amino acids located in the virus membrane"}};

/** Structure defining an atom as it is stored in a PDB file
 */
struct Atom
{
    size_t index;
    size_t serial;
    std::string name;
    std::string altLoc;
    std::string resName;
    std::string chainId;
    size_t reqSeq;
    std::string iCode;
    brayns::Vector3f position;
    float occupancy;
    float tempFactor;
    std::string element;
    std::string charge;
    float radius;
    size_t materialId;
};

struct Sequence
{
    size_t serNum;
    size_t numRes;
    std::vector<std::string> resNames;
};

struct AminoAcid
{
    std::string name;
    std::string shortName;
    //    std::string sideChainClass;
    //    std::string sideChainPolarity;
    //    std::string sideChainCharge;
    //    size_t hidropathyIndex;
    //    size_t absorbance;
    //    float molarAttenuationCoefficient;
    //    float molecularMass;
    //    float occurenceInProtein;
    //    std::string standardGeneticCode;
};

typedef std::vector<Atom> Atoms;
typedef std::set<std::string> Residues;
typedef std::map<size_t, Sequence> SequenceMap;
typedef std::map<std::string, AminoAcid> AminoAcidMap;

/** Loads protein from PDB files
 * http://www.rcsb.org
 */
class AdvancedProteinLoader : public brayns::Loader
{
public:
    AdvancedProteinLoader(brayns::Scene& scene,
                          const brayns::PropertyMap& properties);
    AdvancedProteinLoader(brayns::Scene& scene,
                          const brayns::GeometryParameters& params);

    std::vector<std::string> getSupportedExtensions() const final;
    std::string getName() const final;
    brayns::PropertyMap getProperties() const final;

    bool isSupported(const std::string& filename,
                     const std::string& extension) const final;
    brayns::ModelDescriptorPtr importFromFile(
        const std::string& fileName, const brayns::LoaderProgress& callback,
        const brayns::PropertyMap& properties) const final;

    brayns::ModelDescriptorPtr importFromBlob(
        brayns::Blob&&, const brayns::LoaderProgress&,
        const brayns::PropertyMap&) const final
    {
        throw std::runtime_error("Loading from blob not supported");
    }

private:
    void readAtom(const std::string& line, const ColorScheme colorScheme,
                  const float radiusMultiplier, Atoms& atoms,
                  Residues& residues) const;

    void readSequence(const std::string& line, SequenceMap& sequenceMap) const;

    brayns::PropertyMap _defaults;
};

#endif // ADVANCEDPROTEINLOADER_H
