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
import nrrd
import numpy as np
from skimage import measure
from scipy import ndimage
import blue_brain_atlas_web_exporter.TreeIndexer as TreeIndexer
import blue_brain_atlas_web_exporter
from blue_brain_atlas_web_exporter import __version__


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
        "--out-mesh-dir",
        dest="out_mesh_dir",
        required=True,
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


    return parser.parse_args(args)


def mask_to_mesh_data(arr):
    # dilated = ndimage.binary_dilation(arr, iterations = 1).astype(np.float32)
    gaussian_blurred = ndimage.gaussian_filter(arr - 0.5, sigma=1.5)

    # Make sure the final mesh has no side-of-box hole
    gaussian_blurred[:, :, 0] = -0.5
    gaussian_blurred[:, :, -1] = -0.5
    gaussian_blurred[:, 0, :] = -0.5
    gaussian_blurred[:, -1, :] = -0.5
    gaussian_blurred[0, :, :] = -0.5
    gaussian_blurred[-1, :, :] = -0.5

    vertices, triangles, normals, values = measure.marching_cubes(gaussian_blurred)
    return (vertices, triangles, normals)


def export_obj(vertices, triangles, normals, filepath, origin, transform_3x3, decimation = None):
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

    for v in normals:
        v_x = v[0]
        v_y = v[1]
        v_z = v[2]
        v_x_world = v_x * M_xa + v_y * M_xb + v_z * M_xc + O_x
        v_y_world = v_x * M_ya + v_y * M_yb + v_z * M_yc + O_y
        v_z_world = v_x * M_za + v_y * M_zb + v_z * M_zc + O_z
        obj_str += "n "+str(v_x_world)+" "+str(v_y_world)+" "+str(v_z_world)+" \n"

    for t in triangles:
        obj_str += "f "+str(int(t[2])+1)+" "+str(int(t[1])+1)+" "+str(int(t[0])+1)+" \n"

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
    os.chmod(full_binary_path, 750)
    args = f"{full_binary_path} {filepath} {filepath} {str(decimation)}"
    subprocess.run(args, shell=True, check=True)
    

def main():
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(sys.argv[1:])

    hierarchy = args.hierarchy
    parcellation_volume = args.parcellation_volume
    out_mesh_dir = args.out_mesh_dir
    out_mask_dir = args.out_mask_dir
    out_metadata_path = args.out_metadata

    # create out_mesh_dir if inexistant
    try:
        os.makedirs(out_mesh_dir)
        os.makedirs(out_mask_dir)
    except FileExistsError as e:
        pass

    # reading the nrrd file
    nrrd_data, nrrd_header = nrrd.read(parcellation_volume)
    mask_header = nrrd_header.copy()
    mask_header["type"] = "uint8"

    origin = [0,0,0]
    if "space origin" in nrrd_header:
        origin = nrrd_header["space origin"]
    print(origin)
    
    # As said in the doc (http://teem.sourceforge.net/nrrd/format.html#spacedirections),
    # each vector in "space directions" is for an axis of the array, hence, they are column vectors
    # if the transform were to be represented as 3x3 matrix.
        
    transform_3x3 = [[25, 0, 0],[0, 25, 0], [0, 0, 25]]
    if "space directions" in nrrd_header:
        transform_3x3 = nrrd_header["space directions"]
    print(transform_3x3)

    voxel_world_volume = np.linalg.norm(transform_3x3[0]) * np.linalg.norm(transform_3x3[1]) * np.linalg.norm(transform_3x3[2])
        
    # volume of the whole brain in cubic micrometers
    whole_brain_volume = float(np.count_nonzero(nrrd_data) * voxel_world_volume)

    # loading json annotation
    jsoncontent = json.loads(open(hierarchy, "r").read())

    # sometimes, the 1.json has its content in a "msg" sub prop (the original vesion has).
    # and some other versions don't. Here we deal with both
    if "msg" in jsoncontent:
        flat_tree = TreeIndexer.flattenTree(jsoncontent['msg'][0])
    else:
        flat_tree = TreeIndexer.flattenTree(jsoncontent)

    total_region = len(flat_tree)
    region_counter = 0
    unique_values_in_nrrd = np.unique(nrrd_data)

    # For each region, we create a mask that contains all the sub regions
    metadata = {}
    for region_id in flat_tree:
        region_counter += 1
        region_node = flat_tree[region_id]

        rough_mesh_filepath = os.path.join(out_mesh_dir, str(region_id) + ".obj")
        if os.path.exists(rough_mesh_filepath):
            continue

        print("{}/{} - [{}] {}".format(region_counter, total_region, region_id, flat_tree[region_id]["name"]))

        # region_mask = np.zeros_like(nrrd_data, dtype = "uint8")

        # # masking the current region
        # region_mask[nrrd_data == region_id] = 1
        # subregion_counter = 0
        # total_subregions = len(region_node["_descendants"])

        # print(region_node["_descendants"])

        # for child_id in region_node["_descendants"]:
        #     subregion_counter += 1
        #     print("Grouping subregions {}/{}".format(subregion_counter, total_subregions), end="\r")
        #     if child_id not in unique_values_in_nrrd:
        #         continue
        #     # masking the current region
        #     region_mask[nrrd_data == child_id] = 1

        # all the regions to be added, in theory (aka. not taking into account that some may not be represented in the parcellation volume)
        regions_to_add = region_node["_descendants"] + [region_id]
        
        # list of descendants that are actually represented in the parcellation volume
        represented_regions_to_add = set()
        
        # among all the regions that should be added (in theory), keep only the ones that are actually represented in the annotation volume
        # (this is to speedup thing and not waste time on aggregating not-existing regions)
        for r_id in regions_to_add:
            if r_id in unique_values_in_nrrd:
                represented_regions_to_add.add(r_id)

        if len(represented_regions_to_add) == 0:
            print("Not represented in the annotation volume.")
            continue
        else:
            print("Aggregating regions...")

        def is_in_descendants(val):
            return +(val in represented_regions_to_add)
        
        vectorized_is_in_descendants = np.vectorize(is_in_descendants, otypes = ["uint8"])
        region_mask = vectorized_is_in_descendants(nrrd_data)

        print()

        # if the mask is all black, then there is no mesh to build
        if not np.any(region_mask):
            continue

        # exporting mask files
        print("Export NRRD mask...")
        nrrd.write(os.path.join(out_mask_dir, str(region_id) + ".nrrd"), region_mask, mask_header)

        # Creating the mesh with the marching cube
        print("Marching cube...")
        vertices, triangles, normals = mask_to_mesh_data(region_mask)
    
        # Exporting the mesh as OBJ file
        print("Export OBJ mesh...")
        export_obj(vertices, triangles, normals, rough_mesh_filepath, origin, transform_3x3, decimation=0.15)

        # exporting metadata
        print("Export JSON metadata...")
        region_volume = float(np.count_nonzero(region_mask) * voxel_world_volume)
        metadata[str(region_id)] = {
            "id": region_id,
            "regionVolume": region_volume,
            "regionVolumeRatioToWholeBrain": region_volume / whole_brain_volume,
        }
        
        
    metadata_file = open(out_metadata_path, 'w')
    metadata_file.write(json.dumps(metadata, ensure_ascii = False, indent = 2))
    metadata_file.close()


if __name__ == "__main__":
    main()