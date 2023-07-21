import os
from pathlib import Path

import pooch


def fetch_image_series_example_data(save_directory):
    """Download the sample images for the image series annotator.
    """
    save_directory = Path(save_directory)
    os.makedirs(save_directory, exist_ok=True)
    print("Example data directory is:", save_directory.resolve())
    fname = "image-series.zip"
    unpack_filenames = [os.path.join("series", f"im{i}.tif") for i in range(3)]
    unpack = pooch.Unzip(members=unpack_filenames)
    pooch.retrieve(
        url="https://owncloud.gwdg.de/index.php/s/M1zGnfkulWoAhUG/download",
        known_hash="92346ca9770bcaf55248efee590718d54c7135b6ebca15d669f3b77b6afc8706",
        fname=fname,
        path=save_directory,
        progressbar=True,
        processor=unpack,
    )
    data_folder = os.path.join(save_directory, f"{fname}.unzip", "series")
    assert os.path.exists(data_folder)
    return data_folder


def fetch_wholeslide_example_data(save_directory):
    """Download the sample data for the 2d annotator.

    This downloads part of a whole-slide image from the NeurIPS Cell Segmentation Challenge.
    See https://neurips22-cellseg.grand-challenge.org/ for details on the data.
    """
    save_directory = Path(save_directory)
    os.makedirs(save_directory, exist_ok=True)
    print("Example data directory is:", save_directory.resolve())
    fname = "whole-slide-example-image.tif"
    pooch.retrieve(
        url="https://owncloud.gwdg.de/index.php/s/6ozPtgBmAAJC1di/download",
        known_hash="3ddb9c9dcc844429932ab951eb0743d5a1af83ee9b0ab54f06ceb2090a606d36",
        fname=fname,
        path=save_directory,
        progressbar=True,
    )
    return os.path.join(save_directory, fname)


def fetch_livecell_example_data(save_directory):
    """Download the sample data for the 2d annotator.

    This downloads a single image from the LiveCELL dataset.
    See https://doi.org/10.1038/s41592-021-01249-6 for details on the data.
    """
    save_directory = Path(save_directory)
    os.makedirs(save_directory, exist_ok=True)
    print("Example data directory is:", save_directory.resolve())
    fname = "livecell-2d-image.png"
    pooch.retrieve(
        url="https://owncloud.gwdg.de/index.php/s/fSaOJIOYjmFBjPM/download",
        known_hash="4f190983ea672fc333ac26d735d9625d5abb6e4a02bd4d32523127977a31e8fe",
        fname=fname,
        path=save_directory,
        progressbar=True,
    )
    return os.path.join(save_directory, fname)


def fetch_hela_2d_example_data(save_directory):
    """Download the sample data for the 2d annotator.

    This downloads a single image from the HeLa CTC dataset.
    """
    save_directory = Path(save_directory)
    os.makedirs(save_directory, exist_ok=True)
    print("Example data directory is:", save_directory.resolve())
    fname = "hela-2d-image.png"
    pooch.retrieve(
        url="https://owncloud.gwdg.de/index.php/s/2sr1DHQ34tV7WEb/download",
        known_hash="908fa00e4b273610aa4e0a9c0f22dfa64a524970852f387908f3fa65238259c7",
        fname=fname,
        path=save_directory,
        progressbar=True,
    )
    return os.path.join(save_directory, fname)


def fetch_3d_example_data(save_directory):
    """Download the sample data for the 3d annotator.

    This downloads the Lucchi++ datasets from https://casser.io/connectomics/.
    It is a dataset for mitochondria segmentation in EM.
    """
    save_directory = Path(save_directory)
    os.makedirs(save_directory, exist_ok=True)
    print("Example data directory is:", save_directory.resolve())
    unpack_filenames = [os.path.join("Lucchi++", "Test_In", f"mask{str(i).zfill(4)}.png") for i in range(165)]
    unpack = pooch.Unzip(members=unpack_filenames)
    fname = "lucchi_pp.zip"
    pooch.retrieve(
        url="http://www.casser.io/files/lucchi_pp.zip",
        known_hash="770ce9e98fc6f29c1b1a250c637e6c5125f2b5f1260e5a7687b55a79e2e8844d",
        fname=fname,
        path=save_directory,
        progressbar=True,
        processor=unpack,
    )
    lucchi_dir = save_directory.joinpath(f"{fname}.unzip", "Lucchi++", "Test_In")
    return str(lucchi_dir)


def fetch_tracking_example_data(save_directory):
    """Download the sample data for the tracking annotator.

    This data is the cell tracking challenge dataset DIC-C2DH-HeLa.
    Cell tracking challenge webpage: http://data.celltrackingchallenge.net
    HeLa cells on a flat glass
    Dr. G. van Cappellen. Erasmus Medical Center, Rotterdam, The Netherlands
    Training dataset: http://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip (37 MB)
    Challenge dataset: http://data.celltrackingchallenge.net/challenge-datasets/DIC-C2DH-HeLa.zip (41 MB)
    """
    save_directory = Path(save_directory)
    os.makedirs(save_directory, exist_ok=True)
    print("Example data directory is:", save_directory.resolve())
    unpack_filenames = [os.path.join("DIC-C2DH-HeLa", "01", f"t{str(i).zfill(3)}.tif") for i in range(84)]
    unpack = pooch.Unzip(members=unpack_filenames)
    fname = "DIC-C2DH-HeLa.zip"
    pooch.retrieve(
        url="http://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip",  # 37 MB
        known_hash="fac24746fa0ad5ddf6f27044c785edef36bfa39f7917da4ad79730a7748787af",
        fname=fname,
        path=save_directory,
        progressbar=True,
        processor=unpack,
    )
    cell_tracking_dir = save_directory.joinpath(f"{fname}.unzip", "DIC-C2DH-HeLa", "01")
    assert os.path.exists(cell_tracking_dir)
    return str(cell_tracking_dir)