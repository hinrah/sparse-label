import json

import numpy as np
import nibabel as nib

from constants import Labels
from cross_section import CrossSection
from labeling_strategies import LabelDistantVoxelsAsUnknown, LabelVoxelsBasedOnProjectionToCrossSection, LabelUnknownCrosssSectionBackground, LabelCenterline
from mask_image import SparseMaskImage, join_mask_images

from networkx.readwrite import json_graph


def load_centerline(path):
    with open(path, "r") as file:
        data = json.load(file)
    return json_graph.node_link_graph(data, link="edges")


if __name__ == '__main__':
    all_masks = []
    for i in range(7):
        contour_points_wall = np.load(f"C:\\Users\\hinrich\\Desktop\\data\\test_contour_{i*2+1}.npy")
        contour_points_lumen = np.load(f"C:\\Users\\hinrich\\Desktop\\data\\test_contour_{i*2+2}.npy")

        contour_points_wall[:, :2] *= -1  # transform to RAS space, should be done by user... or optional
        contour_points_lumen[:, :2] *= -1  # transform to RAS space, should be done by user... or optional

        cross_section = CrossSection(contour_points_lumen, contour_points_wall)
        centerline = load_centerline("C:\\Users\\hinrich\\Desktop\\data\\centerline_graph.json")

        image = nib.load("C:\\Users\\hinrich\\Desktop\\data\\test_image.nii.gz")

        mask = SparseMaskImage(image.shape, image.affine)
        label_non_plane_voxels = LabelDistantVoxelsAsUnknown(cross_section, 0.3)
        label_in_plane_voxels = LabelVoxelsBasedOnProjectionToCrossSection(cross_section)
        label_background_as_unknown_based_on_centerline = LabelUnknownCrosssSectionBackground(cross_section, centerline)
        mask.label_points_with_label(label_non_plane_voxels, Labels.UNPROCESSED)
        mask.label_points_with_label(label_in_plane_voxels, Labels.UNPROCESSED)
        mask.label_points_with_label(label_background_as_unknown_based_on_centerline, Labels.BACKGROUND)

        all_masks.append(mask)

    out_mask = join_mask_images(all_masks)
    label_lumen_based_on_centerline = LabelCenterline(centerline, 1, Labels.LUMEN)
    label_background_based_on_centerline = LabelCenterline(centerline, 12.5, Labels.BACKGROUND)
    out_mask.label_points_with_label(label_lumen_based_on_centerline, Labels.UNKNOWN)
    out_mask.label_points_with_label(label_background_based_on_centerline, Labels.UNKNOWN)



    out_image = nib.Nifti1Image(np.astype(out_mask.mask, np.int16), image.affine)
    nib.save(out_image, "C:\\Users\\hinrich\\Desktop\\data\\test_mask.nii.gz")