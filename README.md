# rotalign - rotate volume movies to match landmarks
---

## Background
*rotalign* rotates 3D-movies (.tif) to standardize the position of three points. The three points **CM**, **PB**, and **SA** are supplied in an separate coordinates (.csv). Units are in microns.

First, the volume of each time frame is resampled to isotropic resolution. The pixel sizes are taken from the ImageJ metadata saved inside the .tif file. Then, the volume is shifted, such that **CM** is in the center of the volume. The volume is rotated such that the plane containing **CM**, **PB**, and **SA** is the central Z-plane and **PB** is pointing up in the Y-axis. 

The coordinates file name needs to contain the header column names:

```python
"Frame #",     # frame index (starting from 1)
"name",        # name of the .tif image (relative to coordinate file)
"Xcm",         # X position of CM in micron 
"Ycm",         # ...
"Zcm",         # ...
"Xpb",         # X position of PB in micron
"Ypb",         # ...
"Zpb",         # ...
"Xsa",         # X position of SA in micron
"Ysa",         # ...
"Zsa"          # ...
```

"Frame #" may contain two positions for **SA** for a single time frame. In that case **SA** is set as the mean location of the two.

The ouput movie is writting as .tif file using the suffix **"_aligned.tif"**. Only time frames contained in the coordinates file are processed and exported.

## Usage
Process two .tif files
```bash
# first activate conda environment by
# conda activate rotalign_env
# (see below)

rotalign --coords coordinate-file.csv movie_1.tif movie_2.tif
```

Get CLI help
```
rotalign --help

usage: rotalign [-h] -c COORDS input_tifs [input_tifs ...]

Rotational alignment

positional arguments:
  input_tifs            Input tif file(s)

optional arguments:
  -h, --help            show this help message and exit
  -c COORDS, --coords COORDS
                        Tracked coordinates file for SA, PB and CM (.csv)

```





## Setup

```bash
# open an anaconda/miniconda prompt

# create separate python environment 
conda create -n rotalign_env python=3.9 pip

# activate environment
conda activate rotalign_env

# install lastest version from github
pip install -U git+https://github.com/sommerc/rotalign.git
```
