"""
Automated instance segmentation functionality.
The classes implemented here extend the automatic instance segmentation from Segment Anything:
https://computational-cell-analytics.github.io/micro-sam/micro_sam.html
"""

import os
import pickle
import warnings
from abc import ABC
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import segment_anything.utils.amg as amg_utils
import vigra

from nifty.tools import blocking
from segment_anything.predictor import SamPredictor

from skimage.measure import regionprops
from torchvision.ops.boxes import batched_nms, box_area

from torch_em.model import UNETR
from torch_em.util.segmentation import watershed_from_center_and_boundary_distances

try:
    from napari.utils import progress as tqdm
except ImportError:
    from tqdm import tqdm

from . import util
from ._vendored import batched_mask_to_box, mask_to_rle_pytorch

#
# Utility Functionality
#


class _FakeInput:
    def __init__(self, shape):
        self.shape = shape

    def __getitem__(self, index):
        block_shape = tuple(ind.stop - ind.start for ind in index)
        return np.zeros(block_shape, dtype="float32")


def mask_data_to_segmentation(
    masks: List[Dict[str, Any]],
    with_background: bool,
    min_object_size: int = 0,
    max_object_size: Optional[int] = None,
) -> np.ndarray:
    """Convert the output of the automatic mask generation to an instance segmentation.

    Args:
        masks: The outputs generated by AutomaticMaskGenerator or EmbeddingMaskGenerator.
            Only supports output_mode=binary_mask.
        with_background: Whether the segmentation has background. If yes this function assures that the largest
            object in the output will be mapped to zero (the background value).
        min_object_size: The minimal size of an object in pixels.
        max_object_size: The maximal size of an object in pixels.
    Returns:
        The instance segmentation.
    """

    masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    # we could also get the shape from the crop box
    shape = next(iter(masks))["segmentation"].shape
    segmentation = np.zeros(shape, dtype="uint32")

    def require_numpy(mask):
        return mask.cpu().numpy() if torch.is_tensor(mask) else mask

    seg_id = 1
    for mask in masks:
        if mask["area"] < min_object_size:
            continue
        if max_object_size is not None and mask["area"] > max_object_size:
            continue

        this_seg_id = mask.get("seg_id", seg_id)
        segmentation[require_numpy(mask["segmentation"])] = this_seg_id
        seg_id = this_seg_id + 1

    seg_ids, sizes = np.unique(segmentation, return_counts=True)

    # In some cases objects may be smaller than peviously calculated,
    # since they are covered by other objects. We ensure these also get
    # filtered out here.
    filter_ids = seg_ids[sizes < min_object_size]

    # If we run segmentation with background we also map the largest segment
    # (the most likely background object) to zero. This is often zero already,
    # but it does not hurt to reset that to zero either.
    if with_background:
        bg_id = seg_ids[np.argmax(sizes)]
        filter_ids = np.concatenate([filter_ids, [bg_id]])

    segmentation[np.isin(segmentation, filter_ids)] = 0
    vigra.analysis.relabelConsecutive(segmentation, out=segmentation)

    return segmentation


#
# Classes for automatic instance segmentation
#


class AMGBase(ABC):
    """Base class for the automatic mask generators.
    """
    def __init__(self):
        # the state that has to be computed by the 'initialize' method of the child classes
        self._is_initialized = False
        self._crop_list = None
        self._crop_boxes = None
        self._original_size = None

    @property
    def is_initialized(self):
        """Whether the mask generator has already been initialized.
        """
        return self._is_initialized

    @property
    def crop_list(self):
        """The list of mask data after initialization.
        """
        return self._crop_list

    @property
    def crop_boxes(self):
        """The list of crop boxes.
        """
        return self._crop_boxes

    @property
    def original_size(self):
        """The original image size.
        """
        return self._original_size

    def _postprocess_batch(
        self,
        data,
        crop_box,
        original_size,
        pred_iou_thresh,
        stability_score_thresh,
        box_nms_thresh,
    ):
        orig_h, orig_w = original_size

        # filter by predicted IoU
        if pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > pred_iou_thresh
            data.filter(keep_mask)

        # filter by stability score
        if stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= stability_score_thresh
            data.filter(keep_mask)

        # filter boxes that touch crop boundaries
        keep_mask = ~amg_utils.is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # return to the original image frame
        data["boxes"] = amg_utils.uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])
        # the data from embedding based segmentation doesn't have the points
        # so we skip if the corresponding key can't be found
        try:
            data["points"] = amg_utils.uncrop_points(data["points"], crop_box)
        except KeyError:
            pass

        return data

    def _postprocess_small_regions(self, mask_data, min_area, nms_thresh):

        if len(mask_data["rles"]) == 0:
            return mask_data

        # filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = amg_utils.rle_to_mask(rle)

            mask, changed = amg_utils.remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = amg_utils.remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask, dtype=torch.int).unsqueeze(0))
            # give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores, dtype=torch.float),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                # mask_data["rles"][i_mask] = amg_utils.mask_to_rle_pytorch(mask_torch)[0]
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data

    def _postprocess_masks(self, mask_data, min_mask_region_area, box_nms_thresh, crop_nms_thresh, output_mode):
        # filter small disconnected regions and holes in masks
        if min_mask_region_area > 0:
            mask_data = self._postprocess_small_regions(
                mask_data,
                min_mask_region_area,
                max(box_nms_thresh, crop_nms_thresh),
            )

        # encode masks
        if output_mode == "coco_rle":
            mask_data["segmentations"] = [amg_utils.coco_encode_rle(rle) for rle in mask_data["rles"]]
        elif output_mode == "binary_mask":
            mask_data["segmentations"] = [amg_utils.rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": amg_utils.area_from_rle(mask_data["rles"][idx]),
                "bbox": amg_utils.box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": amg_utils.box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            # the data from embedding based segmentation doesn't have the points
            # so we skip if the corresponding key can't be found
            try:
                ann["point_coords"] = [mask_data["points"][idx].tolist()]
            except KeyError:
                pass
            curr_anns.append(ann)

        return curr_anns

    def _to_mask_data(self, masks, iou_preds, crop_box, original_size, points=None):
        orig_h, orig_w = original_size

        # serialize predictions and store in MaskData
        data = amg_utils.MaskData(masks=masks.flatten(0, 1), iou_preds=iou_preds.flatten(0, 1))
        if points is not None:
            data["points"] = torch.as_tensor(points.repeat(masks.shape[1], axis=0), dtype=torch.float)

        del masks

        # calculate the stability scores
        data["stability_score"] = amg_utils.calculate_stability_score(
            data["masks"], self._predictor.model.mask_threshold, self._stability_score_offset
        )

        # threshold masks and calculate boxes
        data["masks"] = data["masks"] > self._predictor.model.mask_threshold
        data["masks"] = data["masks"].type(torch.bool)
        data["boxes"] = batched_mask_to_box(data["masks"])

        # compress to RLE
        data["masks"] = amg_utils.uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        # data["rles"] = amg_utils.mask_to_rle_pytorch(data["masks"])
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data

    def get_state(self) -> Dict[str, Any]:
        """Get the initialized state of the mask generator.

        Returns:
            State of the mask generator.
        """
        if not self.is_initialized:
            raise RuntimeError("The state has not been computed yet. Call initialize first.")

        return {"crop_list": self.crop_list, "crop_boxes": self.crop_boxes, "original_size": self.original_size}

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set the state of the mask generator.

        Args:
            state: The state of the mask generator, e.g. from serialized state.
        """
        self._crop_list = state["crop_list"]
        self._crop_boxes = state["crop_boxes"]
        self._original_size = state["original_size"]
        self._is_initialized = True

    def clear_state(self):
        """Clear the state of the mask generator.
        """
        self._crop_list = None
        self._crop_boxes = None
        self._original_size = None
        self._is_initialized = False


class AutomaticMaskGenerator(AMGBase):
    """Generates an instance segmentation without prompts, using a point grid.

    This class implements the same logic as
    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py
    It decouples the computationally expensive steps of generating masks from the cheap post-processing operation
    to filter these masks to enable grid search and interactively changing the post-processing.

    Use this class as follows:
    ```python
    amg = AutomaticMaskGenerator(predictor)
    amg.initialize(image)  # Initialize the masks, this takes care of all expensive computations.
    masks = amg.generate(pred_iou_thresh=0.8)  # Generate the masks. This is fast and enables testing parameters
    ```

    Args:
        predictor: The segment anything predictor.
        points_per_side: The number of points to be sampled along one side of the image.
            If None, `point_grids` must provide explicit point sampling.
        points_per_batch: The number of points run simultaneously by the model.
            Higher numbers may be faster but use more GPU memory.
        crop_n_layers: If >0, the mask prediction will be run again on crops of the image.
        crop_overlap_ratio: Sets the degree to which crops overlap.
        crop_n_points_downscale_factor: How the number of points is downsampled when predicting with crops.
        point_grids: A lisst over explicit grids of points used for sampling masks.
            Normalized to [0, 1] with respect to the image coordinate system.
        stability_score_offset: The amount to shift the cutoff when calculating the stability score.
    """
    def __init__(
        self,
        predictor: SamPredictor,
        points_per_side: Optional[int] = 32,
        points_per_batch: Optional[int] = None,
        crop_n_layers: int = 0,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        stability_score_offset: float = 1.0,
    ):
        super().__init__()

        if points_per_side is not None:
            self.point_grids = amg_utils.build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None or not None.")

        self._predictor = predictor
        self._points_per_side = points_per_side

        # we set the points per batch to 16 for mps for performance reasons
        # and otherwise keep them at the default of 64
        if points_per_batch is None:
            points_per_batch = 16 if str(predictor.device) == "mps" else 64
        self._points_per_batch = points_per_batch

        self._crop_n_layers = crop_n_layers
        self._crop_overlap_ratio = crop_overlap_ratio
        self._crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self._stability_score_offset = stability_score_offset

    def _process_batch(self, points, im_size, crop_box, original_size):
        # run model on this batch
        transformed_points = self._predictor.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self._predictor.device, dtype=torch.float)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        masks, iou_preds, _ = self._predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=True,
            return_logits=True,
        )
        data = self._to_mask_data(masks, iou_preds, crop_box, original_size, points=points)
        del masks
        return data

    def _process_crop(self, image, crop_box, crop_layer_idx, verbose, precomputed_embeddings):
        # crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]

        if not precomputed_embeddings:
            self._predictor.set_image(cropped_im)

        # get the points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # generate masks for this crop in batches
        data = amg_utils.MaskData()
        n_batches = len(points_for_image) // self._points_per_batch +\
            int(len(points_for_image) % self._points_per_batch != 0)
        for (points,) in tqdm(
            amg_utils.batch_iterator(self._points_per_batch, points_for_image),
            disable=not verbose, total=n_batches,
            desc="Predict masks for point grid prompts",
        ):
            batch_data = self._process_batch(points, cropped_im_size, crop_box, self.original_size)
            data.cat(batch_data)
            del batch_data

        if not precomputed_embeddings:
            self._predictor.reset_image()

        return data

    @torch.no_grad()
    def initialize(
        self,
        image: np.ndarray,
        image_embeddings: Optional[util.ImageEmbeddings] = None,
        i: Optional[int] = None,
        verbose: bool = False
    ) -> None:
        """Initialize image embeddings and masks for an image.

        Args:
            image: The input image, volume or timeseries.
            image_embeddings: Optional precomputed image embeddings.
                See `util.precompute_image_embeddings` for details.
            i: Index for the image data. Required if `image` has three spatial dimensions
                or a time dimension and two spatial dimensions.
            verbose: Whether to print computation progress.
        """
        original_size = image.shape[:2]
        self._original_size = original_size

        crop_boxes, layer_idxs = amg_utils.generate_crop_boxes(
            original_size, self._crop_n_layers, self._crop_overlap_ratio
        )

        # we can set fixed image embeddings if we only have a single crop box
        # (which is the default setting)
        # otherwise we have to recompute the embeddings for each crop and can't precompute
        if len(crop_boxes) == 1:
            if image_embeddings is None:
                image_embeddings = util.precompute_image_embeddings(self._predictor, image)
            util.set_precomputed(self._predictor, image_embeddings, i=i)
            precomputed_embeddings = True
        else:
            precomputed_embeddings = False

        # we need to cast to the image representation that is compatible with SAM
        image = util._to_image(image)

        crop_list = []
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(
                image, crop_box, layer_idx, verbose=verbose, precomputed_embeddings=precomputed_embeddings
            )
            crop_list.append(crop_data)

        self._is_initialized = True
        self._crop_list = crop_list
        self._crop_boxes = crop_boxes

    @torch.no_grad()
    def generate(
        self,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        box_nms_thresh: float = 0.7,
        crop_nms_thresh: float = 0.7,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
    ) -> List[Dict[str, Any]]:
        """Generate instance segmentation for the currently initialized image.

        Args:
            pred_iou_thresh: Filter threshold in [0, 1], using the mask quality predicted by the model.
            stability_score_thresh: Filter threshold in [0, 1], using the stability of the mask
                under changes to the cutoff used to binarize the model prediction.
            box_nms_thresh: The IoU threshold used by nonmax suppression to filter duplicate masks.
            crop_nms_thresh: The IoU threshold used by nonmax suppression to filter duplicate masks between crops.
            min_mask_region_area: Minimal size for the predicted masks.
            output_mode: The form masks are returned in.

        Returns:
            The instance segmentation masks.
        """
        if not self.is_initialized:
            raise RuntimeError("AutomaticMaskGenerator has not been initialized. Call initialize first.")

        data = amg_utils.MaskData()
        for data_, crop_box in zip(self.crop_list, self.crop_boxes):
            crop_data = self._postprocess_batch(
                data=deepcopy(data_),
                crop_box=crop_box, original_size=self.original_size,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                box_nms_thresh=box_nms_thresh
            )
            data.cat(crop_data)

        if len(self.crop_boxes) > 1 and len(data["crop_boxes"]) > 0:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        data.to_numpy()
        masks = self._postprocess_masks(data, min_mask_region_area, box_nms_thresh, crop_nms_thresh, output_mode)
        return masks


def _compute_tiled_embeddings(predictor, image, image_embeddings, embedding_save_path, tile_shape, halo):
    have_tiling_params = (tile_shape is not None) and (halo is not None)
    if image_embeddings is None and have_tiling_params:
        if embedding_save_path is None:
            raise ValueError(
                "You have passed neither pre-computed embeddings nor a path for saving embeddings."
                "Embeddings with tiling can only be computed if a save path is given."
            )
        image_embeddings = util.precompute_image_embeddings(
            predictor, image, tile_shape=tile_shape, halo=halo, save_path=embedding_save_path
        )
    elif image_embeddings is None and not have_tiling_params:
        raise ValueError("You passed neither pre-computed embeddings nor tiling parameters (tile_shape and halo)")
    else:
        feats = image_embeddings["features"]
        tile_shape_, halo_ = feats.attrs["tile_shape"], feats.attrs["halo"]
        if have_tiling_params and (
            (list(tile_shape) != list(tile_shape_)) or
            (list(halo) != list(halo_))
        ):
            warnings.warn(
                "You have passed both pre-computed embeddings and tiling parameters (tile_shape and halo) and"
                "the values of the tiling parameters from the embeddings disagree with the ones that were passed."
                "The tiling parameters you have passed wil be ignored."
            )
        tile_shape = tile_shape_
        halo = halo_

    return image_embeddings, tile_shape, halo


class TiledAutomaticMaskGenerator(AutomaticMaskGenerator):
    """Generates an instance segmentation without prompts, using a point grid.

    Implements the same functionality as `AutomaticMaskGenerator` but for tiled embeddings.

    Args:
        predictor: The segment anything predictor.
        points_per_side: The number of points to be sampled along one side of the image.
            If None, `point_grids` must provide explicit point sampling.
        points_per_batch: The number of points run simultaneously by the model.
            Higher numbers may be faster but use more GPU memory.
        point_grids: A lisst over explicit grids of points used for sampling masks.
            Normalized to [0, 1] with respect to the image coordinate system.
        stability_score_offset: The amount to shift the cutoff when calculating the stability score.
    """

    # We only expose the arguments that make sense for the tiled mask generator.
    # Anything related to crops doesn't make sense, because we re-use that functionality
    # for tiling, so these parameters wouldn't have any effect.
    def __init__(
        self,
        predictor: SamPredictor,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        point_grids: Optional[List[np.ndarray]] = None,
        stability_score_offset: float = 1.0,
    ) -> None:
        super().__init__(
            predictor=predictor,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            point_grids=point_grids,
            stability_score_offset=stability_score_offset,
        )

    @torch.no_grad()
    def initialize(
        self,
        image: np.ndarray,
        image_embeddings: Optional[util.ImageEmbeddings] = None,
        i: Optional[int] = None,
        tile_shape: Optional[Tuple[int, int]] = None,
        halo: Optional[Tuple[int, int]] = None,
        verbose: bool = False,
        embedding_save_path: Optional[str] = None,
    ) -> None:
        """Initialize image embeddings and masks for an image.

        Args:
            image: The input image, volume or timeseries.
            image_embeddings: Optional precomputed image embeddings.
                See `util.precompute_image_embeddings` for details.
            i: Index for the image data. Required if `image` has three spatial dimensions
                or a time dimension and two spatial dimensions.
            tile_shape: The tile shape for embedding prediction.
            halo: The overlap of between tiles.
            verbose: Whether to print computation progress.
            embedding_save_path: Where to save the image embeddings.
        """
        original_size = image.shape[:2]
        self._original_size = original_size

        image_embeddings, tile_shape, halo = _compute_tiled_embeddings(
            self._predictor, image, image_embeddings, embedding_save_path, tile_shape, halo
        )

        tiling = blocking([0, 0], original_size, tile_shape)
        n_tiles = tiling.numberOfBlocks

        # the crop box is always the full local tile
        tiles = [tiling.getBlockWithHalo(tile_id, list(halo)).outerBlock for tile_id in range(n_tiles)]
        crop_boxes = [[tile.begin[1], tile.begin[0], tile.end[1], tile.end[0]] for tile in tiles]

        # we need to cast to the image representation that is compatible with SAM
        image = util._to_image(image)

        mask_data = []
        for tile_id in tqdm(range(n_tiles), total=n_tiles, desc="Compute masks for tile", disable=not verbose):
            # set the pre-computed embeddings for this tile
            features = image_embeddings["features"][tile_id]
            tile_embeddings = {
                "features": features,
                "input_size": features.attrs["input_size"],
                "original_size": features.attrs["original_size"],
            }
            util.set_precomputed(self._predictor, tile_embeddings, i)

            # compute the mask data for this tile and append it
            this_mask_data = self._process_crop(
                image, crop_box=crop_boxes[tile_id], crop_layer_idx=0, verbose=verbose, precomputed_embeddings=True
            )
            mask_data.append(this_mask_data)

        # set the initialized data
        self._is_initialized = True
        self._crop_list = mask_data
        self._crop_boxes = crop_boxes


#
# Instance segmentation functionality based on fine-tuned decoder
#


class DecoderAdapter(torch.nn.Module):
    """Adapter to contain the UNETR decoder in a single module.

    To apply the decoder on top of pre-computed embeddings for
    the segmentation functionality.
    See also: https://github.com/constantinpape/torch-em/blob/main/torch_em/model/unetr.py
    """
    def __init__(self, unetr):
        super().__init__()

        self.base = unetr.base
        self.out_conv = unetr.out_conv
        self.deconv_out = unetr.deconv_out
        self.decoder_head = unetr.decoder_head
        self.final_activation = unetr.final_activation
        self.postprocess_masks = unetr.postprocess_masks

        self.decoder = unetr.decoder
        self.deconv1 = unetr.deconv1
        self.deconv2 = unetr.deconv2
        self.deconv3 = unetr.deconv3
        self.deconv4 = unetr.deconv4

    def forward(self, input_, input_shape, original_shape):
        z12 = input_

        z9 = self.deconv1(z12)
        z6 = self.deconv2(z9)
        z3 = self.deconv3(z6)
        z0 = self.deconv4(z3)

        updated_from_encoder = [z9, z6, z3]

        x = self.base(z12)
        x = self.decoder(x, encoder_inputs=updated_from_encoder)
        x = self.deconv_out(x)

        x = torch.cat([x, z0], dim=1)
        x = self.decoder_head(x)

        x = self.out_conv(x)
        if self.final_activation is not None:
            x = self.final_activation(x)

        x = self.postprocess_masks(x, input_shape, original_shape)
        return x


# TODO refactor this once the exact layout for the new model architecture is clear
def get_custom_sam_model_with_decoder(
    checkpoint: Union[os.PathLike, str],
    model_type: str,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    """
    device = util.get_device(device)

    # over-ride the unpickler with our custom one
    custom_pickle = pickle
    custom_pickle.Unpickler = util._CustomUnpickler

    state = torch.load(checkpoint, map_location=device, pickle_module=custom_pickle)

    # Get the predictor.
    model_state = state["model_state"]
    sam_prefix = "sam."
    model_state = OrderedDict(
        [(k[len(sam_prefix):] if k.startswith(sam_prefix) else k, v) for k, v in model_state.items()]
    )

    sam = util.sam_model_registry[model_type]()
    sam.to(device)
    sam.load_state_dict(model_state)
    predictor = SamPredictor(sam)
    predictor.model_type = model_type

    # Get the decoder.
    # NOTE: we hard-code the UNETR settings for now.
    # Eventually we may need to finds a way to be more flexible.
    unetr = UNETR(
        backbone="sam",
        encoder=predictor.model.image_encoder,
        out_channels=3,
        use_sam_stats=True,
        final_activation="Sigmoid",
        use_skip_connection=False,
        resize_input=True,
    )

    encoder_state = []
    encoder_prefix = "image_"
    encoder_state = OrderedDict(
        (k[len(encoder_prefix):], v) for k, v in model_state.items() if k.startswith(encoder_prefix)
    )

    decoder_state = state["decoder_state"]
    unetr_state = OrderedDict(list(encoder_state.items()) + list(decoder_state.items()))
    unetr.load_state_dict(unetr_state)
    unetr.to(device)

    decoder = DecoderAdapter(unetr)

    return predictor, decoder


class InstanceSegmentationWithDecoder:
    """Generates an instance segmentation without prompts, using a decoder.

    Implements the same interface as `AutomaticMaskGenerator`.

    Use this class as follows:
    ```python
    segmenter = InstanceSegmentationWithDecoder(predictor, decoder)
    segmenter.initialize(image)   # Predict the image embeddings and decoder outputs.
    masks = segmenter.generate(center_distance_threshold=0.75)  # Generate the instance segmentation.
    ```

    Args:
        predictor: The segment anything predictor.
        decoder: The decoder to predict intermediate representations
            for instance segmentation.
    """
    def __init__(
        self,
        predictor: SamPredictor,
        decoder: torch.nn.Module,
    ) -> None:
        self._predictor = predictor
        self._decoder = decoder

        # The decoder outputs.
        self._foreground = None
        self._center_distances = None
        self._boundary_distances = None

        self._is_initialized = False

    @property
    def is_initialized(self):
        """Whether the mask generator has already been initialized.
        """
        return self._is_initialized

    @torch.no_grad()
    def initialize(
        self,
        image: np.ndarray,
        image_embeddings: Optional[util.ImageEmbeddings] = None,
        i: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize image embeddings and decoder predictions for an image.

        Args:
            image: The input image, volume or timeseries.
            image_embeddings: Optional precomputed image embeddings.
                See `util.precompute_image_embeddings` for details.
            i: Index for the image data. Required if `image` has three spatial dimensions
                or a time dimension and two spatial dimensions.
            verbose: Dummy input to be compatible with other function signatures.
        """
        if image_embeddings is None:
            image_embeddings = util.precompute_image_embeddings(self._predictor, image)

        # This could be made more versatile to also support other decoder inputs,
        # e.g. the UNETR with skip connections.
        if isinstance(image_embeddings["features"], torch.Tensor):
            embeddings = image_embeddings["features"]
        else:
            embeddings = torch.from_numpy(image_embeddings["features"])

        if i is not None:
            embeddings = embeddings[i]
        embeddings = embeddings.to(self._predictor.device)

        input_shape = tuple(image_embeddings["input_size"])
        original_shape = tuple(image_embeddings["original_size"])
        output = self._decoder(
            embeddings, input_shape, original_shape
        ).cpu().numpy().squeeze(0)

        assert output.shape[0] == 3, f"{output.shape}"

        self._foreground = output[0]
        self._center_distances = output[1]
        self._boundary_distances = output[2]

        self._is_initialized = True

    def _to_masks(self, segmentation, output_mode):
        if output_mode != "binary_mask":
            raise NotImplementedError

        props = regionprops(segmentation)
        ndim = segmentation.ndim
        assert ndim in (2, 3)

        shape = segmentation.shape
        if ndim == 2:
            crop_box = [0, shape[1], 0, shape[0]]
        else:
            crop_box = [0, shape[2], 0, shape[1], 0, shape[0]]

        # go from skimage bbox in format [y0, x0, y1, x1] to SAM format [x0, w, y0, h]
        def to_bbox_2d(bbox):
            y0, x0 = bbox[0], bbox[1]
            w = bbox[3] - x0
            h = bbox[2] - y0
            return [x0, w, y0, h]

        def to_bbox_3d(bbox):
            z0, y0, x0 = bbox[0], bbox[1], bbox[2]
            w = bbox[5] - x0
            h = bbox[4] - y0
            d = bbox[3] - y0
            return [x0, w, y0, h, z0, d]

        to_bbox = to_bbox_2d if ndim == 2 else to_bbox_3d
        masks = [
            {
                "segmentation": segmentation == prop.label,
                "area": prop.area,
                "bbox": to_bbox(prop.bbox),
                "crop_box": crop_box,
                "seg_id": prop.label,
            } for prop in props
        ]
        return masks

    # TODO find good default values (empirically)
    def generate(
        self,
        center_distance_threshold: float = 0.5,
        boundary_distance_threshold: float = 0.5,
        foreground_threshold: float = 0.5,
        foreground_smoothing: float = 0.75,
        distance_smoothing: float = 1.6,
        min_size: int = 0,
        output_mode: Optional[str] = "binary_mask",
    ) -> List[Dict[str, Any]]:
        """Generate instance segmentation for the currently initialized image.

        Args:
            center_distance_threshold: Center distance predictions below this value will be
                used to find seeds (intersected with thresholded boundary distance predictions).
            boundary_distance_threshold: Boundary distance predictions below this value will be
                used to find seeds (intersected with thresholded center distance predictions).
            foreground_smoothing: Sigma value for smoothing the foreground predictions, to avoid
                checkerboard artifacts in the prediction.
            foreground_threshold: Foreground predictions above this value will be used as foreground mask.
            distance_smoothing: Sigma value for smoothing the distance predictions.
            min_size: Minimal object size in the segmentation result.
            output_mode: The form masks are returned in. Pass None to directly return the instance segmentation.

        Returns:
            The instance segmentation masks.
        """
        if not self.is_initialized:
            raise RuntimeError("InstanceSegmentationWithDecoder has not been initialized. Call initialize first.")

        if foreground_smoothing > 0:
            foreground = vigra.filters.gaussianSmoothing(self._foreground, foreground_smoothing)
        else:
            foreground = self._foreground
        segmentation = watershed_from_center_and_boundary_distances(
            self._center_distances, self._boundary_distances, foreground,
            center_distance_threshold=center_distance_threshold,
            boundary_distance_threshold=boundary_distance_threshold,
            foreground_threshold=foreground_threshold,
            distance_smoothing=distance_smoothing,
            min_size=min_size,
        )
        if output_mode is not None:
            segmentation = self._to_masks(segmentation, output_mode)
        return segmentation

    def get_state(self) -> Dict[str, Any]:
        """Get the initialized state of the instance segmenter.

        Returns:
            Instance segmentation state.
        """
        if not self.is_initialized:
            raise RuntimeError("The state has not been computed yet. Call initialize first.")

        return {
            "foreground": self._foreground,
            "center_distances": self._center_distances,
            "boundary_distances": self._boundary_distances,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set the state of the instance segmenter.

        Args:
            state: The instance segmentation state
        """
        self._foreground = state["foreground"]
        self._center_distances = state["center_distances"]
        self._boundary_distances = state["boundary_distances"]
        self._is_initialized = True

    def clear_state(self):
        """Clear the state of the instance segmenter.
        """
        self._foreground = None
        self._center_distances = None
        self._boundary_distances = None
        self._is_initialized = False


def get_amg(
    predictor: SamPredictor,
    is_tiled: bool,
    decoder: Optional[torch.nn.Module] = None,
    **kwargs,
) -> Union[AMGBase, InstanceSegmentationWithDecoder]:
    """Get the automatic mask generator class.

    Args:
        predictor: The segment anything predictor.
        is_tiled: Whether tiled embeddings are used.
        decoder: Decoder to predict instacne segmmentation.
        kwargs: The keyword arguments for the amg class.

    Returns:
        The automatic mask generator.
    """
    if decoder is None:
        segmenter = TiledAutomaticMaskGenerator(predictor, **kwargs) if is_tiled else\
            AutomaticMaskGenerator(predictor, **kwargs)
    else:
        if is_tiled:
            raise NotImplementedError
        segmenter = InstanceSegmentationWithDecoder(predictor, decoder, **kwargs)
    return segmenter
