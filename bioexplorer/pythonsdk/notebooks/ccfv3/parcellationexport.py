# Copyright 2020 - 2023 Blue Brain Project / EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import sys
import json
import os
import platform
import subprocess
import re
import nrrd
import numpy as np
from skimage import measure
from scipy import ndimage

import blue_brain_atlas_web_exporter
from blue_brain_atlas_web_exporter import __version__
import blue_brain_atlas_web_exporter.TreeIndexer as TreeIndexer
import blue_brain_atlas_web_exporter.json_to_jsonld as json_to_jsonld

from multiprocessing import Pool, cpu_count

descendants = TreeIndexer.DESCENDANTS_PROP_NAME
children = json_to_jsonld.CHILDREN
represented = json_to_jsonld.REPRESENTED
regionVolume = json_to_jsonld.REGIONVOLUME
regionVolumeRatio = json_to_jsonld.REGIONVOLUMERATIO

OUT_MESH_DIR = "--out-mesh-dir"
mesh_ext = "obj"


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="From volumetric parcellations exports meshes and binary masks.")
    parser.add_argument(
        "--version",
        action="version",
        version="parcellationexport {ver}".format(ver=__version__))

    parser.add_argument(
        "--hierarchy",
        dest="hierarchy",
        required=True,
        metavar="<FILE PATH>",
        help="The hierarchy JSON file, sometimes called 1.json")

    parser.add_argument(
        "--parcellation-volume",
        dest="parcellation_volume",
        required=True,
        metavar="<FILE PATH>",
        help="The NRRD parcellation volume file")

    parser.add_argument(
        OUT_MESH_DIR,
        dest="out_mesh_dir",
        required=False,
        metavar="<DIRECTORY PATH>",
        help="The output directory to create the OBJ region mesh files")

    parser.add_argument(
        "--out-mask-dir",
        dest="out_mask_dir",
        required=True,
        metavar="<DIRECTORY PATH>",
        help="The output directory to create the NRRD region mask files")

    parser.add_argument(
        "--out-metadata",
        dest="out_metadata",
        required=True,
        metavar="<FILE PATH>",
        help="Path to the output metadata file (json)")

    parser.add_argument(
        "--out-hierarchy-volume",
        dest="out_hierarchy_volume",
        required=True,
        metavar="<FILE PATH>",
        help="Path to the output hierarchy including volume info (json)")
    
    parser.add_argument(
        "--out-hierarchy-jsonld",
        dest="out_hierarchy_jsonld",
        required=True,
        metavar="<FILE PATH>",
        help="Path to the output hierarchy JSON-LD file built from the input "
        "hierarchy JSON file.")

    return parser.parse_args(args)


def voxel_world_volume(transform_3x3):
    # As said in the doc (http://teem.sourceforge.net/nrrd/format.html#spacedirections),
    # each vector in "space directions" is for an axis of the array, hence, they are column vectors
    # if the transform were to be represented as 3x3 matrix.
    return np.linalg.norm(transform_3x3[0]) * np.linalg.norm(transform_3x3[1]) * np.linalg.norm(transform_3x3[2])


def mask_to_mesh_data(arr):
    dilated = ndimage.binary_dilation(arr, iterations=1).astype(np.float32)
    gaussian_blurred = ndimage.gaussian_filter(dilated - 0.5, sigma=3)

    # Make sure the final mesh has no side-of-box hole
    gaussian_blurred[:, :, 0] = -0.5
    gaussian_blurred[:, :, -1] = -0.5
    gaussian_blurred[:, 0, :] = -0.5
    gaussian_blurred[:, -1, :] = -0.5
    gaussian_blurred[0, :, :] = -0.5
    gaussian_blurred[-1, :, :] = -0.5

    vertices, triangles, normals, values = measure.marching_cubes(gaussian_blurred)
    return vertices, triangles


def export_obj(vertices, triangles, filepath, origin, transform_3x3, decimation=None):
    """
        | xa  xb  xc |
    M = | ya  yb  yc |  --> M is transform_3x3
        | za  zb  zc |

        | x |
    O = | y |  --> O is origin
        | z |
    
    """

    M_xa = transform_3x3[0][0]
    M_ya = transform_3x3[0][1]
    M_za = transform_3x3[0][2]

    M_xb = transform_3x3[1][0]
    M_yb = transform_3x3[1][1]
    M_zb = transform_3x3[1][2]

    M_xc = transform_3x3[2][0]
    M_yc = transform_3x3[2][1]
    M_zc = transform_3x3[2][2]

    O_x = origin[0]
    O_y = origin[1]
    O_z = origin[2]

    obj_str = ""

    for v in vertices:
        v_x = v[0]
        v_y = v[1]
        v_z = v[2]
        v_x_world = v_x * M_xa + v_y * M_xb + v_z * M_xc + O_x
        v_y_world = v_x * M_ya + v_y * M_yb + v_z * M_yc + O_y
        v_z_world = v_x * M_za + v_y * M_zb + v_z * M_zc + O_z
        obj_str += "v "+str(v_x_world)+" "+str(v_y_world)+" "+str(v_z_world)+" \n"

    for t in triangles:
        obj_str += "f "+str(int(t[2])+1)+" "+str(int(t[1])+1)+" "+str(int(t[0])+1)+" \n"
        # f.write("f "+str(int(t[0])+1)+" "+str(int(t[1])+1)+" "+str(int(t[2])+1)+" \n")

    f = open(filepath, 'w')
    f.write(obj_str)
    f.close()

    if not decimation:
        return
    
    # from here, we are using the binaries to reduce the number of vertices and faces of the mesh
    module_dirpath = os.path.dirname(blue_brain_atlas_web_exporter.__file__)
    os_to_dir = {
        "Linux": os.path.join("bin.Linux", "simplify"),
        "Darwin": os.path.join("bin.OSX", "simplify"),
        "Windows": os.path.join("bin.Windows", "simplify.exe"),
    }

    full_binary_path = os.path.join(module_dirpath, "simplify", os_to_dir[platform.system()])
    try:
        os.chmod(full_binary_path, 750)
    except OSError:
        pass
    args = f"{full_binary_path} {filepath} {filepath} {str(decimation)}"
    subprocess.run(args, shell=True, check=True)


def nodeToLayerIndex(node):
    """
    From a Node (TreeIndexer), extract the list of cortical layers
    """
    upper_acronym = node['acronym'].upper()
    cerebral_cortex_id = 688

    if cerebral_cortex_id not in node[TreeIndexer.ASCENDANTS_PROP_NAME]:
        return []

    if upper_acronym.startswith("CA"):
        return []

    # find the first digit in the acronym
    m = re.search(r"-\d", upper_acronym)
    if not m:
        return []
    
    first_digit_index = m.start() + 1  # 1 accounts for the dash

    # the layer term can help us spot the regions that are in a different "column" but same layer
    layer_term_upper = upper_acronym[first_digit_index:]

    # Example: "1" or "2"
    if len(layer_term_upper) == 1:
        return [layer_term_upper]

    # Example: "6A" or "6B". In this particular case, we want to return "6A" but also "6"
    if len(layer_term_upper) == 2:
        return [layer_term_upper, layer_term_upper[0]]

    # Example: "2/3". This pattern is never used for more than 2 layers so we won't find "1/3"
    if "/" in layer_term_upper:
        return layer_term_upper.split('/')

    # Example: sometimes for more than 2 layers "1-3", or Sometimes used for only two "1-2"
    if "-" in layer_term_upper:
        top_layer, bottom_layer = list(map(int, layer_term_upper.split('-')))
        all_layers = list(map(str, list(range(top_layer, bottom_layer + 1))))
        return all_layers


def getNeighboursIds(roi_mask, whole_brain_parcellation):
    # Create a mask of the roi outer contour and apply it to the whole brain volume
    dilated_roi = ndimage.binary_dilation(roi_mask, iterations=1).astype(np.uint32)
    roi_neighbour_mask = dilated_roi - roi_mask
    neighbour_regions = whole_brain_parcellation.copy()
    neighbour_regions[roi_neighbour_mask == 0] = 0

    contour_nb_voxels = np.count_nonzero(roi_neighbour_mask)

    # List the neighbour regions by their ids
    neighbour_regions_ids, neighbour_regions_counts = np.unique(neighbour_regions, return_counts=True)
    neighbour_regions_ratios = neighbour_regions_counts / contour_nb_voxels
    adjacent_to_ratios = dict(zip(neighbour_regions_ids.tolist(), neighbour_regions_ratios.tolist()))
    del adjacent_to_ratios[0]
    return adjacent_to_ratios


def writeMetadata(data, filepath):
    metadata_file = open(filepath, 'w')
    metadata_file.write(json.dumps(data, ensure_ascii=False, indent=2))
    metadata_file.close()


def check_leaves_only(regions_in_annotation, flat_tree, children_key):
    annotation_regions_not_in_hierarchy = []
    non_leaf_hierarchy_regions_in_annotation = []
    for reg_id in regions_in_annotation:
        if reg_id == 0:
            continue
        if reg_id in flat_tree:
            # A leaf region has an empty 'children_key' list
            if flat_tree[reg_id][children_key]:
                non_leaf_hierarchy_regions_in_annotation.append(reg_id)
        else:
            annotation_regions_not_in_hierarchy.append(reg_id)
    if annotation_regions_not_in_hierarchy or non_leaf_hierarchy_regions_in_annotation:
        msg = f"{len(annotation_regions_not_in_hierarchy)} regions in the annotation volume " \
            f"were not found in the hierarchy:\n{annotation_regions_not_in_hierarchy}\n"
        msg += f"{len(non_leaf_hierarchy_regions_in_annotation)} regions in the annotation volume " \
            f"were found being not leaves in the hierarchy:\n{non_leaf_hierarchy_regions_in_annotation}"
        raise Exception(msg)


def get_flat_tree(hierarchy_path, children_key):
    jsoncontent = json.loads(open(hierarchy_path, "r").read())
    if "msg" in jsoncontent:
        jsoncontent_body = jsoncontent['msg'][0]
    else:
        jsoncontent_body = jsoncontent
    return TreeIndexer.flattenTree(jsoncontent_body, children_prop_name=children_key), jsoncontent_body, jsoncontent


def get_neighbour_regions(region_id, adjacent_counts, region_layers_set, parcellation_volume, flat_tree, hierarchy):
    """
    Get the list of region IDs that are neighbors to the specified region.

    Parameters:
    - region_id (int): The ID of the target region.
    - adjacent_counts (dict): A dictionary mapping neighboring region IDs to their voxel counts.
    - region_layers_set (set): A set of layer indices associated with the target region.
    - parcellation_volume (str): Path to the parcellation volume NRRD file.
    - flat_tree (dict): The flattened representation of the brain region hierarchy.
    - hierarchy (str): Path to the hierarchy JSON file.

    Returns:
    - list: A list of region IDs that are neighbors to the target region and share at least one layer.
    """
    continuous_with = []

    for nei_id in adjacent_counts:
        if nei_id not in flat_tree:
            print(f"Region {nei_id} is adjacent to {region_id} in {parcellation_volume}, but it is not present in {hierarchy}")
            continue
        nei_node = flat_tree[nei_id]
        nei_layers = set(nodeToLayerIndex(nei_node))
        # check if this neighbour has some layers in common with the ROI
        if not nei_layers.isdisjoint(region_layers_set):
            continuous_with.append(nei_id)

    return continuous_with


def get_unique_values_in_nrrd(nrrd_file):
    nrrd_data, _ = nrrd.read(nrrd_file)
    return np.unique(nrrd_data)


def main_(hierarchy, parcellation_volume, out_mask_dir, out_mesh_dir, out_metadata_path,
          out_hierarchy_volume_path, out_hierarchy_jsonld_path):
    if out_mesh_dir:
        os.makedirs(out_mesh_dir, exist_ok=True)
    else:
        print("No %s provided, meshes will not be exported" % OUT_MESH_DIR)
    os.makedirs(out_mask_dir, exist_ok=True)

    # reading the nrrd file
    nrrd_data, nrrd_header = nrrd.read(parcellation_volume)
    mask_header = nrrd_header.copy()
    mask_header["type"] = "uint8"

    volume_unit = "cubic micrometer"  # missing in the nrrd_header, need to hardcode

    transform_3x3 = nrrd_header["space directions"]
    # volume of the whole brain in cubic micrometers
    voxel_volume = voxel_world_volume(transform_3x3)
    whole_brain_volume = float(np.count_nonzero(nrrd_data) * voxel_volume)

    # loading json annotation
    flat_tree, jsoncontent_body, jsoncontent = get_flat_tree(hierarchy, children)
    jsoncontent_body["unitCode"] = volume_unit

    total_region = len(flat_tree)
    unique_values_in_nrrd = get_unique_values_in_nrrd(parcellation_volume)

    # For each region, we create a mask that contains all the sub-regions
    async_result = {}

    pool = Pool(processes=cpu_count() - 2)
    for region_counter, region_id in enumerate(flat_tree):
        region_node = flat_tree[region_id]

        print(f"\nProcessing region {region_counter}/{total_region}: [{region_id}] '{region_node['name']}'")
        async_result[region_id] = pool.apply_async(process_region, args=(region_id, region_node, flat_tree, hierarchy,
            parcellation_volume, nrrd_data, unique_values_in_nrrd, voxel_volume, whole_brain_volume, volume_unit,
            nrrd_header.copy(), transform_3x3, out_mask_dir, out_mesh_dir))
    pool.close()
    pool.join()

    metadata = {}
    keys_to_remove = ["atlas_id", "graph_order", "st_level"]
    represented_regions = []
    for region_id in flat_tree:
        region_id_str = str(region_id)
        metadata[region_id_str] = async_result[region_id].get()
        if metadata[region_id_str][represented]:
            represented_regions.append(region_id)

        # Find region to update in the hierarchy
        jsoncontent_region = jsoncontent_body
        while jsoncontent_region["id"] != region_id:
            for jsoncontent_region_ch in jsoncontent_region[children]:
                if jsoncontent_region_ch["id"] == region_id:
                    jsoncontent_region = jsoncontent_region_ch
                    break
                else:
                    if region_id not in flat_tree[jsoncontent_region_ch["id"]][descendants]:
                        continue
                    else:
                        jsoncontent_region = jsoncontent_region_ch

        # Update region
        jsoncontent_region.update(metadata[region_id_str])
        # Remove keys not present in the new regions from the leaves-only hierarchy, to keep uniform regions dictionary
        for key in keys_to_remove:
            jsoncontent_region.pop(key, None)

    # exporting the metadata for the whole brain
    writeMetadata(metadata, out_metadata_path)

    # exporting a new hierarchy JSON including the regions info
    writeMetadata(jsoncontent, out_hierarchy_volume_path)

    if out_hierarchy_jsonld_path:
        # transforming the hierarchy JSON to JSONLD
        hierarchy_jsonld = json_to_jsonld.hierarchy_json_to_jsonld(jsoncontent_body)
        if hierarchy_jsonld:
            writeMetadata(hierarchy_jsonld, out_hierarchy_jsonld_path)
        else:
            raise Exception("Failed to generate a JSONLD version of the hierarchy JSON file")

    if out_mesh_dir:  # check that all the meshes have been created
        for region_id in represented_regions:
            mesh_name = ".".join([str(region_id), mesh_ext])
            if mesh_name not in os.listdir(out_mesh_dir):
                raise Exception(f"Region {region_id} is represented in {parcellation_volume} but no mesh {mesh_name} is"
                                f" present in {out_mesh_dir}")
        print(f"\nA mesh is available (in {out_mesh_dir}) for any region represented in {parcellation_volume}")


def process_region(region_id, region_node, flat_tree, hierarchy, parcellation_volume, nrrd_data, unique_values_in_nrrd,
    voxel_volume, whole_brain_volume, volume_unit, mask_header, transform_3x3, out_mask_dir, out_mesh_dir):
    """
    Process a specific brain region and export its metadata, mask, and mesh if applicable.

    Args:
        region_id (int): The ID of the brain region.
        region_node (dict): The node information of the brain region from the hierarchy.
        flat_tree (dict): The flattened representation of the hierarchy tree.
        hierarchy (str): The path to the hierarchy JSON file.
        parcellation_volume (str): The path to the parcellation volume NRRD file.
        nrrd_data (numpy.ndarray): The NRRD data representing the parcellation volume.
        unique_values_in_nrrd (numpy.ndarray): Unique values present in the parcellation volume.
        voxel_volume (float): The volume of a single voxel in cubic micrometers.
        whole_brain_volume (float): The volume of the entire brain in cubic micrometers.
        volume_unit (str): The unit of volume measurement.
        mask_header (dict): The header information for the mask.
        transform_3x3 (numpy.ndarray): The 3x3 transformation matrix for voxel-to-world mapping.
        out_mask_dir (str): The directory to export the region masks.
        out_mesh_dir (str): The directory to export the region meshes.

    Returns:
        dict: Metadata information for the processed region.
    """
    # getting the list of layers for this region
    region_layers = nodeToLayerIndex(region_node)
    region_layers_set = set(region_layers)

    # if the mask is all black, then there is no mesh to build,
    # though we still want to list the region in metadata
    metadata_reg = {
        "id": region_id,
        represented: False,
        "unitCode": volume_unit,
        regionVolume: None,
        regionVolumeRatio: None,
        "layers": region_layers,
        "adjacentTo": None,
        "continuousWith": None,
    }

    # all the regions (not taking into account that some may not be represented in the parcellation volume)
    regions_to_add = region_node[descendants] + [region_id]
    # list of descendants that are actually represented in the parcellation volume:
    # among all the regions, keep only those that are actually represented in the annotation volume
    # (this is to speedup thing and not waste time on aggregating not-existing regions)
    represented_regions_to_add = set(r_id for r_id in regions_to_add if r_id in unique_values_in_nrrd)

    if len(represented_regions_to_add) == 0:
        print(f"Region not represented in the annotation volume, nor its {descendants}.")
        return metadata_reg
    else:
        metadata_reg[represented] = True

    print("Aggregating regions...")

    def is_in_descendants(val):
        return int(val in represented_regions_to_add)

    vectorized_is_in_descendants = np.vectorize(is_in_descendants, otypes=["uint8"])
    region_mask = vectorized_is_in_descendants(nrrd_data)

    # computing region neighbours
    print("Computing adjacency...")
    adjacent_counts = getNeighboursIds(region_mask, nrrd_data)
    continuous_with = get_neighbour_regions(region_id, adjacent_counts, region_layers_set, parcellation_volume, flat_tree, hierarchy)

    # Updating metadata
    print("Add JSON metadata...")
    region_volume = float(np.count_nonzero(region_mask) * voxel_volume)
    metadata_reg[regionVolume] = region_volume
    metadata_reg[regionVolumeRatio] = region_volume / whole_brain_volume
    metadata_reg["adjacentTo"] = adjacent_counts
    metadata_reg["continuousWith"] = continuous_with

    # exporting mask files
    print("Export NRRD mask...")
    nrrd.write(os.path.join(out_mask_dir, f"{region_id}.nrrd"), region_mask,
               mask_header)
    # Exporting metadata for this current brain region
    writeMetadata(metadata_reg, os.path.join(out_mask_dir, f"{region_id}.json"))

    if out_mesh_dir:  # much longer than previous steps
        # Creating the mesh with the marching cube
        print("Marching cube...")
        # vertices, triangles, normals, values = measure.marching_cubes_lewiner(region_mask)
        vertices, triangles = mask_to_mesh_data(region_mask)
        # Exporting the mesh as OBJ file
        print("Export OBJ mesh...")
        rough_mesh_filepath = os.path.join(out_mesh_dir, f"{region_id}.{mesh_ext}")
        # export_obj(vertices, triangles, rough_mesh_filepath, origin, transform_3x3, origin, transform_3x3)
        export_obj(vertices, triangles, rough_mesh_filepath, mask_header["space origin"], transform_3x3,
                   decimation=0.15)

    return metadata_reg


def main():
    """Main entry point allowing external calls
    """
    args = parse_args(sys.argv[1:])

    unique_values_in_annotation = get_unique_values_in_nrrd(args.parcellation_volume)
    flat_tree, _, _ = get_flat_tree(args.hierarchy, children)

    print("Performing check for leaves-only annotation")
    check_leaves_only(unique_values_in_annotation, flat_tree, children)

    main_(args.hierarchy, args.parcellation_volume, args.out_mask_dir, args.out_mesh_dir,
          args.out_metadata, args.out_hierarchy_volume, args.out_hierarchy_jsonld)
