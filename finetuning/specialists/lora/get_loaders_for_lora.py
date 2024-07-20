import os

import numpy as np
from math import ceil, floor

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import light_microscopy, electron_microscopy
from torch_em.transform.label import PerObjectDistanceTransform
from torch_em.transform.raw import normalize_percentile, normalize
import micro_sam.training as sam_training
from micro_sam.training.util import ResizeLabelTrafo, ResizeRawTrafo


ROOT = "/scratch/usr/nimcarot/data"

class RawTrafo:
    def __init__(self, desired_shape=None, do_padding=True, do_rescaling=False, padding="constant", triplicate_dims=False):
        self.desired_shape = desired_shape
        self.padding = padding
        self.do_rescaling = do_rescaling
        self.triplicate_dims = triplicate_dims
        self.do_padding = do_padding

    def __call__(self, raw):
        if self.do_rescaling:
            if len(raw.shape) == 3:
                raw = normalize_percentile(raw, axis=(1, 2))
                raw = np.mean(raw, axis=0)
                raw = normalize(raw)
                raw = raw * 255
            else:
                raw = normalize(raw)
                raw = raw * 255

        if self.do_padding:
            assert self.desired_shape is not None
            #print("Raw Shape:")
            #print(raw.shape)
            tmp_ddim = (self.desired_shape[-2] - raw.shape[-2], self.desired_shape[-1] - raw.shape[-1])
            ddim = (tmp_ddim[0] / 2, tmp_ddim[1] / 2)
            raw = np.pad(
                raw,
                pad_width=((ceil(ddim[0]), floor(ddim[0])), (ceil(ddim[1]), floor(ddim[1]))),
                mode=self.padding
            )   
            #print(f"Afer padding:{raw.shape}")
            assert raw.shape[-2:] == self.desired_shape[-2:], raw.shape
        
        if self.triplicate_dims:
            if raw.ndim == 3 and raw.shape[0] == 1:
                raw = np.concatenate((raw, raw, raw), axis=0)
            if raw.ndim == 2: 
                raw = np.stack((raw, raw, raw), axis = 0)

            #print(f"Raw Shape after triplicate_dims:{raw.shape}")


        return raw


class LabelTrafo:
    def __init__(self, desired_shape=None, padding="constant", min_size=0, triplicate_dims=False, do_padding=True):
        self.desired_shape = desired_shape
        self.padding = padding
        self.min_size = min_size
        self.triplicate_dims = triplicate_dims
        self.do_padding = do_padding

    def __call__(self, labels):
        if labels.ndim == 3:
            assert labels.shape[0] == 1
            labels = labels[0]

        if self.triplicate_dims:
            if labels.ndim == 2: 
                labels = np.stack(labels, axis = 0)

        distance_trafo = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False,
            foreground=True, instances=True, min_size=self.min_size
        )

        labels = distance_trafo(labels)

        if self.do_padding:
            # choosing H and W from labels (4, H, W), from above dist trafo outputs
            assert self.desired_shape is not None
            tmp_ddim = (self.desired_shape[0] - labels.shape[1], self.desired_shape[1] - labels.shape[2])
            ddim = (tmp_ddim[0] / 2, tmp_ddim[1] / 2)
            #print("labels shape:")
            #print(labels.shape)
            labels = np.pad(
                labels,
                pad_width=((0,0), (ceil(ddim[0]), floor(ddim[0])), (ceil(ddim[1]), floor(ddim[1]))),
                mode=self.padding
            )
            #print(labels.shape)
            assert labels.shape[1:] == self.desired_shape, labels.shape

        return labels


def _fetch_loaders(dataset_name):

    if dataset_name == "covid_if":
        # 1, Covid IF does not have internal splits. For this example I chose first 10 samples for training,
        # and next 3 samples for validation, left the rest for testing.

        label_transform = PerObjectDistanceTransform(distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True, min_size=0)
        raw_transform = RawTrafo(do_padding=False, do_rescaling=True)

        train_loader = light_microscopy.get_covid_if_loader(
            path=os.path.join(ROOT, "covid_if"),
            patch_shape=(512, 512),
            batch_size=2,
            sample_range=(None, 10),
            target="cells",
            num_workers=16,
            shuffle=True,
            download=True,
            raw_transform=raw_transform,
            label_transform=label_transform
        )
        val_loader = light_microscopy.get_covid_if_loader(
            path=os.path.join(ROOT, "covid_if"),
            patch_shape=(512, 512),
            batch_size=1,
            sample_range=(10, 13),
            target="cells",
            num_workers=16,
            download=True,
            raw_transform=raw_transform,
            label_transform=label_transform,
        )

    elif dataset_name == "orgasegment":
        # 2. OrgaSegment has internal splits provided. We follow the respective splits for our experiments.
        
        raw_transform = RawTrafo(do_padding=False, triplicate_dims=True)
        label_transform = LabelTrafo(do_padding=False, triplicate_dims=True)

        train_loader = light_microscopy.get_orgasegment_loader(
            path=os.path.join(ROOT, "orgasegment"),
            patch_shape=(512, 512),
            split="train",
            batch_size=2,
            num_workers=16,
            shuffle=True,
            download=True,
            raw_transform=raw_transform,
            label_transform=label_transform,
            sampler=MinInstanceSampler()
        )
        val_loader = light_microscopy.get_orgasegment_loader(
            path=os.path.join(ROOT, "orgasegment"),
            patch_shape=(512, 512),
            split="val",
            batch_size=1,
            num_workers=16,
            download=True,
            raw_transform=raw_transform,
            label_transform=label_transform,
            sampler=MinInstanceSampler()
        )

    elif dataset_name == "mouse-embryo":
        # 3. Mouse Embryo
        # the logic used here is: I use the first 100 slices per volume from the training split for training
        # and the next ~20/30 slices per volume from the training split for validation
        # and we use the whole volume from the val set for testing
        train_rois = [np.s_[0:100, :, :], np.s_[0:100, :, :], np.s_[0:100, :, :], np.s_[0:100, :, :]]
        val_rois = [np.s_[100:, :, :], np.s_[100:, :, :], np.s_[100:, :, :], np.s_[100:, :, :]]

        raw_transform = RawTrafo((1,512,512), do_rescaling=True)
        label_transform = LabelTrafo((512,512))

        train_loader = light_microscopy.get_mouse_embryo_loader(
            path=os.path.join(ROOT, "mouse-embryo"),
            name="membrane",
            split="train",
            patch_shape=(1, 512, 512),
            batch_size=2,
            download=True,
            num_workers=16,
            shuffle=True,
            sampler=MinInstanceSampler(min_num_instances=3),
            rois=train_rois,
            raw_transform=raw_transform,
            label_transform=label_transform,
            ndim=2
        )
        val_loader = light_microscopy.get_mouse_embryo_loader(
            path=os.path.join(ROOT, "mouse-embryo"),
            name="membrane",
            split="train",
            patch_shape=(1, 512, 512),
            batch_size=1,
            download=True,
            num_workers=16,
            sampler=MinInstanceSampler(min_num_instances=3),
            rois=val_rois,
            raw_transform=raw_transform,
            label_transform=label_transform,
            ndim=2
        )

    elif dataset_name == "mitolab_glycolytic_muscle":
        # 4. This dataset would need aspera-cli to be installed, I'll provide you with this data
        # ...
        train_rois = np.s_[0:175, :, :]
        val_rois = np.s_[175:225, :, :]
        test_rois = np.s_[225:, :, :]

        raw_transform = RawTrafo((1,512,512))
        label_transform = LabelTrafo((512,512))

        train_loader = electron_microscopy.cem.get_benchmark_loader(
            path=os.path.join(ROOT, "mitolab"),
            dataset_id=3,
            batch_size=2,
            patch_shape=(1, 512, 512),
            download=False,
            num_workers=16,
            shuffle=True,
            sampler=MinInstanceSampler(),
            rois=train_rois,
            raw_transform=raw_transform,
            label_transform=label_transform,
            ndim=2
        )
        val_loader = electron_microscopy.cem.get_benchmark_loader(
            path=os.path.join(ROOT, "mitolab"),
            dataset_id=3,
            batch_size=2,
            patch_shape=(1, 512, 512),
            download=False,
            num_workers=16,
            shuffle=True,
            sampler=MinInstanceSampler(),
            rois=val_rois,
            raw_transform=raw_transform,
            label_transform=label_transform,
            ndim=2
        )

    elif dataset_name == "platy_cilia":
        # 5. Platynereis (Cilia)
        # the logic used here is: I use the first 85 slices per volume from the training split for training
        # and the next ~10-15 slices per volume from the training split for validation
        # and we use the whole volume from the val set for testing
        train_rois = {
            1: np.s_[0:85, :, :], 2: np.s_[0:85, :, :], 3: np.s_[0:85, :, :]
        }
        val_rois = {
            1: np.s_[85:, :, :], 2: np.s_[85:, :, :], 3: np.s_[85:, :, :]
        }

        raw_transform = RawTrafo((3,512,512), triplicate_dims=False)
        label_transform = LabelTrafo((512,512))

        train_loader = electron_microscopy.get_platynereis_cilia_loader(
            path=os.path.join(ROOT, "platynereis"),
            patch_shape=(1, 512, 512),
            ndim=2,
            batch_size=2,
            rois=train_rois,
            download=True,
            num_workers=16,
            shuffle=True,
            sampler=MinInstanceSampler(),
            raw_transform=raw_transform,
            label_transform=label_transform
        )
        val_loader = electron_microscopy.get_platynereis_cilia_loader(
            path=os.path.join(ROOT, "platynereis"),
            patch_shape=(1, 512, 512),
            ndim=2,
            batch_size=1,
            rois=val_rois,
            download=True,
            num_workers=16,
            sampler=MinInstanceSampler(),
            raw_transform=raw_transform,
            label_transform=label_transform
        )

    else:
        raise ValueError(f"{dataset_name} is not a valid dataset name.")

    return train_loader, val_loader


def _verify_loaders():
    dataset_name = "mitolab_glycolytic_muscle"

    train_loader, val_loader = _fetch_loaders(dataset_name=dataset_name)

    breakpoint()

    # NOTE: if using on the cluster, napari visualization won't work with "check_loader".
    # turn "plt=True" and provide path to save the matplotlib outputs of the loader.
    check_loader(train_loader, 8, plt=True, save_path="./train_loader.png")
    check_loader(val_loader, 8, plt=True, save_path="./val_loader.png")


if __name__ == "__main__":
    _verify_loaders()