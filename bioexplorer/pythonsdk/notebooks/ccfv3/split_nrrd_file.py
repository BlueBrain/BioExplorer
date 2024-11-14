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

import SimpleITK as sitk

def cut_nrrd_along_x(input_path, output_path_1, output_path_2):
    # Read the NRRD file
    original_image = sitk.ReadImage(input_path)

    # Get image size
    size = original_image.GetSize()
    print(size)

    # Define the cropping region for the first sub-image along the z-axis
    region_1 = (0, 0, 0, size[0], size[1], size[2] // 2)
    print(region_1)

    # Define the cropping region for the second sub-image along the z-axis
    region_2 = (0, 0, size[2] // 2, size[0], size[1], size[2] - (size[2] // 2))
    print(region_2)

    # Crop sub-images
    sub_image_1 = sitk.Crop(original_image, region_1)
    sub_image_2 = sitk.Crop(original_image, region_2)

    # Save the sub-images
    sitk.WriteImage(sub_image_1, output_path_1)
    sitk.WriteImage(sub_image_2, output_path_2)

# Example usage
input_file = "annotation_25_2022_CCFv3a.nrrd"
output_file_1 = "annotation_25_2022_CCFv3a_a.nrrd"
output_file_2 = "annotation_25_2022_CCFv3a_b.nrrd"

cut_nrrd_along_x(input_file, output_file_1, output_file_2)
