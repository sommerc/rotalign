import os
import numpy as np
import argparse
import h5py
import skimage.transform as tf
import tifffile


def downscale2h5():
    def get_args():
        parser = argparse.ArgumentParser(
            description="Downscale and export to ilastik .h5"
        )

        parser.add_argument(
            "input_tifs",
            type=str,
            help="Input tif file(s)",
            nargs="+",
        )

        parser.add_argument(
            "-c", "--channel", type=int, help="Channel to downscale", default=2
        )

        parser.add_argument("-s", "--scale", type=float, help="Scaling", default=0.5)

        return parser.parse_args()

    args = get_args()

    print("\nDownscale and export to ilastik .h5:")
    print("#" * 80)
    for arg in vars(args):
        print(f" {arg:28s}", getattr(args, arg))
    print("#" * 80)

    for fn in args.input_tifs:
        assert os.path.exists(fn)

        print("Reading input tif")
        img = tifffile.imread(fn)

        img = img[:, :, args.channel, :, :]

        print("Rescaling")
        img_ds = tf.rescale(
            img,
            scale=args.scale,
            channel_axis=0,
            preserve_range=True,
            anti_aliasing=True,
        )

        print("Change dtype")
        img_ds = (img_ds.clip(0, 255) + 0.5).astype(np.uint8)[..., None]

        fn_base, _ = os.path.splitext(fn)
        out_h5_fn = f"{fn_base}.h5"

        print("Export h5")
        with h5py.File(out_h5_fn, "w") as hf:
            hf.create_dataset("data", data=img_ds, chunks=(1, 32, 32, 32, 1))
        print("Done")


if __name__ == "__main__":
    pass
