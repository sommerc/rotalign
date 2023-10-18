import os
import numpy as np
import argparse
import skimage.transform as tf
import tifffile
import pandas as pd
from scipy.ndimage import shift

from tqdm.auto import tqdm, trange

from scipy import ndimage as ndi

from matplotlib import pyplot as plt


def read_tiff_voxel_size_zyx(file_path):
    """
    Implemented based on information found in https://pypi.org/project/tifffile
    """

    def _xy_voxel_size(tags, key):
        assert key in ["XResolution", "YResolution"]
        if key in tags:
            num_pixels, units = tags[key].value
            return units / num_pixels
        # return default
        return 1.0

    with tifffile.TiffFile(file_path) as tiff:
        image_metadata = tiff.imagej_metadata
        if image_metadata is not None:
            z = image_metadata.get("spacing", 1.0)
        else:
            # default voxel size
            z = 1.0

        tags = tiff.pages[0].tags
        # parse X, Y resolution
        y = _xy_voxel_size(tags, "YResolution")
        x = _xy_voxel_size(tags, "XResolution")
        # return voxel size
        return np.array([z, y, x])


def umeyama(P, Q):
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P
    centeredQ = Q

    C = np.dot(np.transpose(centeredP), centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        print("Mirror?")
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    return R


def normvec(vec):
    return vec / np.linalg.norm(vec)


def get_args():
    parser = argparse.ArgumentParser(description="Rotational alignment")
    parser.add_argument(
        "input_tifs",
        type=str,
        help="Input tif file(s)",
        nargs="+",
    )

    parser.add_argument(
        "-c",
        "--coords",
        type=str,
        nargs=1,
        help="Tracked coordinates file for SA, PB and CM (.csv)",
        required=True,
    )

    return parser.parse_args()


REQ_COLUMNS = [
    "Frame #",
    "name",
    "Xcm",
    "Ycm",
    "Zcm",
    "Xpb",
    "Ypb",
    "Zpb",
    "Xsa",
    "Ysa",
    "Zsa",
]


def run(mov_in_fn, coord_fn):
    assert os.path.exists(mov_in_fn), f"Tif file '{mov_in_fn}' not found"
    assert os.path.exists(coord_fn), f"Coord file '{coord_fn}' not found"

    tif_path, tif_fn = os.path.split(mov_in_fn)
    tif_base_fn, _ = os.path.splitext(tif_fn)

    print(f"Reading file {tif_fn}")
    print("*" * 80)
    mov_in_raw = tifffile.imread(mov_in_fn)
    z_pxs, y_pxs, x_pxs = read_tiff_voxel_size_zyx(mov_in_fn)

    COORDS_TAB = pd.read_csv(coord_fn)

    for c in REQ_COLUMNS:
        assert (
            c in COORDS_TAB.columns
        ), f"Column '{c}' not in found in file '{coord_fn}'"

    COORDS_TAB = COORDS_TAB[REQ_COLUMNS]
    COORDS_TAB["frame"] = COORDS_TAB["Frame #"] - 1

    coords_tab = COORDS_TAB[COORDS_TAB["name"] == tif_base_fn].sort_values("frame")
    if len(coords_tab) == 0:
        print(f"Movie {tif_base_fn} not found in coords file {coord_fn}. Skipping...")
        return

    t_min = coords_tab.frame.min()
    t_max = coords_tab.frame.max()
    coords_tab["frame"] = coords_tab["frame"] - t_min

    coords_tab = coords_tab.groupby("frame")[
        ["Xcm", "Ycm", "Zcm", "Xpb", "Ypb", "Zpb", "Xsa", "Ysa", "Zsa"]
    ].mean()

    # crop time
    t_size, z_size, c_size, y_size, x_size = mov_in_raw.shape

    mov_in_raw = mov_in_raw[t_min : t_max + 1, ...]

    t_size = mov_in_raw.shape[0]

    #

    coords_tab["Xsa_px"] = (coords_tab["Xsa"] / x_pxs).astype(int)
    coords_tab["Ysa_px"] = (coords_tab["Ysa"] / y_pxs).astype(int)
    coords_tab["Zsa_px"] = (coords_tab["Zsa"] / z_pxs).astype(int)

    coords_tab["Xcm_px"] = (coords_tab["Xcm"] / x_pxs).astype(int)
    coords_tab["Ycm_px"] = (coords_tab["Ycm"] / y_pxs).astype(int)
    coords_tab["Zcm_px"] = (coords_tab["Zcm"] / z_pxs).astype(int)

    coords_tab["Xpb_px"] = (coords_tab["Xpb"] / x_pxs).astype(int)
    coords_tab["Ypb_px"] = (coords_tab["Ypb"] / y_pxs).astype(int)
    coords_tab["Zpb_px"] = (coords_tab["Zpb"] / z_pxs).astype(int)

    mov_centered = np.zeros((t_size, z_size, c_size, y_size, x_size), dtype=np.uint8)

    for t in trange(t_size, desc="  - centering to CM"):
        for c in range(c_size):
            mov_slice = mov_in_raw[t, :, c, :, :]
            z_sft = z_size // 2 - coords_tab.loc[t].Zcm_px
            y_sft = y_size // 2 - coords_tab.loc[t].Ycm_px
            x_sft = x_size // 2 - coords_tab.loc[t].Xcm_px
            shift(
                mov_slice,
                shift=[z_sft, y_sft, x_sft],
                order=0,
                output=mov_centered[t, :, c, ...],
            )

    z_iso_size = int(round((z_pxs / x_pxs) * z_size))
    mov_iso = np.zeros((t_size, z_iso_size, c_size, y_size, x_size), dtype=np.uint8)
    for t in trange(t_size, desc="  - resampling to isotropic resolution"):
        for c in range(c_size):
            mov_iso[t, :, c, ...] = ndi.zoom(
                mov_centered[t, :, c, ...], (z_pxs / x_pxs, 1, 1), order=1
            )

    mov_out = np.zeros((t_size, z_iso_size, c_size, y_size, x_size), np.uint8)

    for t in trange(t_size, desc="  - Aligning frames...", position=1):
        vec_sa = [
            coords_tab["Zsa"].loc[t] - coords_tab["Zcm"].loc[t],
            coords_tab["Ysa"].loc[t] - coords_tab["Ycm"].loc[t],
            coords_tab["Xsa"].loc[t] - coords_tab["Xcm"].loc[t],
        ]

        vec_pb = [
            coords_tab["Zpb"].loc[t] - coords_tab["Zcm"].loc[t],
            coords_tab["Ypb"].loc[t] - coords_tab["Ycm"].loc[t],
            coords_tab["Xpb"].loc[t] - coords_tab["Xcm"].loc[t],
        ]

        vec_sa_norm = normvec(vec_sa)
        vec_pb_norm = normvec(vec_pb)

        P = np.array([vec_pb_norm, np.cross(vec_sa_norm, vec_pb_norm)])
        Q = np.array([[0, -1, 0], [-1, 0, 0]])

        R = umeyama(P, Q)

        mg = np.mgrid[
            -z_iso_size // 2 : z_iso_size // 2,
            -y_size // 2 : y_size // 2,
            -x_size // 2 : x_size // 2,
        ]
        mg_rot = R.dot(mg.reshape(3, -1)).reshape(mg.shape)

        cc = (mg_rot.T + np.array([z_iso_size // 2, 256, 256])).T

        for c in trange(c_size, desc="  |_ channel...", position=1, leave=False):
            mov_t = tf.warp(mov_iso[t, :, c], cc, preserve_range=True, order=1)
            mov_out[t, :, c, :, :] = mov_t

    print(f"Saving output...")
    print("*" * 80)
    tifffile.imsave(
        f"{tif_path}/{tif_base_fn}_aligned.tif",
        mov_out,
        imagej=True,
        resolution=(1.0 / x_pxs, 1.0 / y_pxs),
        metadata={"spacing": x_pxs, "unit": "um", "axes": "TZCYX"},
    )
    print()


def main():
    args = get_args()

    print("\nRotational Embryo alignh")
    print("#" * 80)
    for arg in vars(args):
        print(f" {arg:28s}", getattr(args, arg))
    print("#" * 80)

    for fn in args.input_tifs:
        run(os.path.abspath(fn), args.coords[0])


if __name__ == "__main__":
    main()
