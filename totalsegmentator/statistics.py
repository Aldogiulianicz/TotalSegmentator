import pathlib
from pathlib import Path
import json
from functools import partial
from typing import Union

import numpy as np
import SimpleITK as sitk
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
from tqdm import tqdm

from totalsegmentator.map_to_binary import class_map


def get_radiomics_features(seg_file, img_file="ct.nii.gz"):

    def _resample(image, spacing_out=(3.0, 3.0, 3.0), is_label=False):
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        new_size = [int(np.ceil(osz * ospc / nspc))
            for osz, ospc, nspc in zip(original_size, original_spacing, spacing_out)]

        interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
        default_val = 0 if is_label else 0.0

        return sitk.Resample(
            image,
            new_size,
            sitk.Transform(),
            interpolator,
            image.GetOrigin(),
            spacing_out,
            image.GetDirection(),
            default_val,
            image.GetPixelID()
        )


    standard_features = [
        'firstorder_energy', 'firstorder_entropy',
        'firstorder_interquartile_range', 'firstorder_kurtosis', 'firstorder_maximum', 'firstorder_mean',
        'firstorder_mean_absolute_deviation', 'firstorder_median', 'firstorder_minimum',
        'firstorder_percentile10', 'firstorder_percentile90', 'firstorder_range',
        'firstorder_robust_mean_absolute_deviation', 'firstorder_root_mean_squared',
        'firstorder_skewness', 'firstorder_standard_deviation', 'firstorder_total_energy',
        'firstorder_uniformity', 'firstorder_variance',
        'glcm_Autocorrelation', 'glcm_ClusterProminence',
        'glcm_ClusterShade', 'glcm_ClusterTendency', 'glcm_Contrast', 'glcm_Correlation',
        'glcm_DifferenceAverage', 'glcm_DifferenceEntropy', 'glcm_DifferenceVariance', 'glcm_Id',
        'glcm_Idm', 'glcm_Idmn', 'glcm_Idn', 'glcm_Imc1', 'glcm_Imc2', 'glcm_InverseVariance',
        'glcm_JointAverage', 'glcm_JointEnergy', 'glcm_JointEntropy', 'glcm_MaximumProbability',
        'glcm_Mcc', 'glcm_SumAverage', 'glcm_SumEntropy', 'glcm_SumSquares',
        'gldm_DependenceEntropy',
        'gldm_DependenceNonUniformity', 'gldm_DependenceNonUniformityNormalized', 'gldm_DependenceVariance',
        'gldm_GrayLevelNonUniformity', 'gldm_GrayLevelVariance', 'gldm_HighGrayLevelEmphasis',
        'gldm_LargeDependenceEmphasis', 'gldm_LargeDependenceHighGrayLevelEmphasis',
        'gldm_LargeDependenceLowGrayLevelEmphasis', 'gldm_LowGrayLevelEmphasis',
        'gldm_SmallDependenceEmphasis', 'gldm_SmallDependenceHighGrayLevelEmphasis',
        'gldm_SmallDependenceLowGrayLevelEmphasis',
        'glrlm_GrayLevelNonUniformity',
        'glrlm_GrayLevelNonUniformityNormalized', 'glrlm_GrayLevelVariance',
        'glrlm_HighGrayLevelRunEmphasis', 'glrlm_LongRunEmphasis', 'glrlm_LongRunHighGrayLevelEmphasis',
        'glrlm_LongRunLowGrayLevelEmphasis', 'glrlm_LowGrayLevelRunEmphasis', 'glrlm_RunEntropy',
        'glrlm_RunLengthNonUniformity', 'glrlm_RunLengthNonUniformityNormalized', 'glrlm_RunPercentage',
        'glrlm_RunVariance', 'glrlm_ShortRunEmphasis', 'glrlm_ShortRunHighGrayLevelEmphasis',
        'glrlm_ShortRunLowGrayLevelEmphasis',
        'glszm_GrayLevelNonUniformity',
        'glszm_GrayLevelNonUniformityNormalized', 'glszm_GrayLevelVariance', 'glszm_HighGrayLevelZoneEmphasis',
        'glszm_LargeAreaEmphasis', 'glszm_LargeAreaHighGrayLevelEmphasis', 'glszm_LargeAreaLowGrayLevelEmphasis',
        'glszm_LowGrayLevelZoneEmphasis', 'glszm_SizeZoneNonUniformity', 'glszm_SizeZoneNonUniformityNormalized',
        'glszm_SmallAreaEmphasis', 'glszm_SmallAreaHighGrayLevelEmphasis', 'glszm_SmallAreaLowGrayLevelEmphasis',
        'glszm_ZoneEntropy', 'glszm_ZonePercentage', 'glszm_ZoneVariance',
        'ngtdm_Busyness', 'ngtdm_Coarseness', 'ngtdm_Complexity', 'ngtdm_Contrast', 'ngtdm_Strength',
        'shape3d_elongation', 'shape3d_flatness', 'shape3d_least_axis_length', 'shape3d_major_axis_length',
        'shape3d_maximum_2d_diameter_column', 'shape3d_maximum_2d_diameter_row',
        'shape3d_maximum_2d_diameter_slice', 'shape3d_maximum_3d_diameter', 'shape3d_mesh_volume',
        'shape3d_minor_axis_length', 'shape3d_number_of_islands', 'shape3d_sphericity',
        'shape3d_surface_area', 'shape3d_surface_volume_ratio', 'shape3d_voxel_volume',
    ]

    try:
        import radiomicsjl as rjl

        mask = sitk.ReadImage(str(seg_file))
        seg = sitk.GetArrayFromImage(mask)

        labels = np.unique(seg)  # unique values present in the mask

        if len(labels) <= 1:
            print("WARNING: Entire mask is 0 or uniform. Setting all features to 0")
            features = {0: {feat: 0 for feat in standard_features}}
        else:
            
            labels = labels[labels != 0].astype(int)  # remove background and convert to int

            image = sitk.ReadImage(str(img_file))

            image_res = _resample(image)
            mask_res = _resample(mask, is_label=True)

            features = {}
            for lab in labels:
                try:
                    f = rjl.extract_radiomic_features(image_res, mask_res, labels = int(lab))
                    features[int(lab)] = {k: round(float(v), 4) for k, v in f.items()}
                except Exception as e:
                    print(f"WARNING: label {lab} skipped ({e})")
                    features[int(lab)] = {feat: 0 for feat in standard_features}
    except Exception as e:
        print(f"WARNING: radiomics raised an exception (setting all features to 0): {e}")
        features = {0: {feat: 0 for feat in standard_features}}

    return seg_file.name.split(".")[0], features

def get_radiomics_features_for_entire_dir(ct_file:Path, mask_dir:Path, file_out:Path):
    masks = sorted(list(mask_dir.glob("*.nii.gz")))
    stats = [get_radiomics_features(mask, ct_file) for mask in masks]
    stats = {mask_name: stats for mask_name, stats in stats}
    with open(file_out, "w") as f:
        json.dump(stats, f, indent=4)


# def is_in_first_or_last_slice(mask):
#     """
#     Check if a mask is in the first or last slice of the image.
#     Then we do not calc any statistics for it because the mask
#     is incomplete.
#     """
#     # in first or last slice segmentation of bad. Therefore they one slice before and after.
#     first_slice = mask[:, :, 1]
#     last_slice = mask[:, :, -2]
#     return (first_slice.sum() > 0) or (last_slice.sum() > 0)


def touches_border(mask):
    """
    Check if mask touches any of the borders. Then we do not calc any statistics for it because the mask
    is incomplete.
    Do not check last two slices but the previous one, because segmentation on last slices often bad.
    """
    if np.any(mask[:3, :, :]) or np.any(mask[-3:, :, :]):
        return True
    if np.any(mask[:, :3, :]) or np.any(mask[:, -3:, :]):
        return True
    if np.any(mask[:, :, :3]) or np.any(mask[:, :, -3:]):
        return True
    return False


def _empty_extra_metrics(target):
    target["n_voxels"] = 0
    target["intensity_std"] = 0.0
    target["intensity_min"] = 0.0
    target["intensity_max"] = 0.0
    target["centroid_vox"] = None
    target["bbox_vox"] = None


def _add_extra_metrics(target, data, ct):
    """Add morphometric + intensity-distribution metrics for one structure.

    centroid_vox and bbox_vox are in voxel/index coordinates (numpy axes i,j,k);
    bbox_vox is [[i_min,i_max],[j_min,j_max],[k_min,k_max]] with inclusive maxima.
    """
    idx = np.argwhere(data)
    if idx.shape[0] == 0:
        _empty_extra_metrics(target)
        return
    vals = ct[data]
    target["n_voxels"] = int(idx.shape[0])
    target["intensity_std"] = float(np.std(vals, dtype=np.float64).round(5))
    target["intensity_min"] = round(float(vals.min()), 5)
    target["intensity_max"] = round(float(vals.max()), 5)
    target["centroid_vox"] = [round(float(c), 2) for c in idx.mean(axis=0)]
    target["bbox_vox"] = [[int(idx[:, a].min()), int(idx[:, a].max())] for a in range(idx.shape[1])]


def get_basic_statistics(seg: np.array,
                         ct_file: Union[Path, Nifti1Image],
                         file_out: Union[Path, None]=None,
                         quiet: bool=False,
                         task: str="total",
                         exclude_masks_at_border: bool=True,
                         roi_subset: list=None,
                         metric: str="mean",
                         normalized_intensities: bool=False,
                         extra_metrics: bool=False):
    """
    ct_file: path to a ct_file or a nifti file object

    extra_metrics: in addition to volume and intensity, also compute
        n_voxels, intensity_std/min/max and the morphometric centroid_vox and
        bbox_vox (both in voxel/index coordinates). Off by default to keep the
        default runtime unchanged.
    """
    ct_img = nib.load(ct_file) if isinstance(ct_file, (str, Path)) else ct_file
    # float32: no truncation of fractional (MR) intensities, no int16 overflow above 32767
    ct = ct_img.get_fdata(dtype=np.float32)
    spacing = ct_img.header.get_zooms()
    vox_vol = spacing[0] * spacing[1] * spacing[2]
    
    if normalized_intensities:
        ct = (ct - ct.min()) / (ct.max() - ct.min())

    class_map_stats = class_map[task]
    if roi_subset is not None:
        class_map_stats = {k: v for k, v in class_map_stats.items() if v in roi_subset}
    
    stats = {}
    for k, mask_name in tqdm(class_map_stats.items(), disable=quiet):
        stats[mask_name] = {}
        # data = nib.load(mask).get_fdata()  # loading: 0.6s
        data = seg == k  # 0.18s
        if touches_border(data) and exclude_masks_at_border:
            # print(f"WARNING: {mask_name} touches border. Skipping.")
            stats[mask_name]["volume"] = 0.0
            stats[mask_name]["intensity"] = 0.0
            if extra_metrics:
                _empty_extra_metrics(stats[mask_name])
        else:
            stats[mask_name]["volume"] = data.sum() * vox_vol  # vol in mm3; 0.2s
            has_voxels = data.any()
            if metric == "mean":
                stats[mask_name]["intensity"] = float(ct[data].mean(dtype=np.float64).round(5)) if has_voxels else 0.0
            elif metric == "median":
                stats[mask_name]["intensity"] = round(float(np.median(ct[data])), 5) if has_voxels else 0.0
            if extra_metrics:
                _add_extra_metrics(stats[mask_name], data, ct)

    if file_out is not None:
        with open(file_out, "w") as f:
            json.dump(stats, f, indent=4)

    return stats
