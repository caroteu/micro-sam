import os
import argparse

import torch

from torch_em.model import UNETR
from torch_em.loss import DiceBasedDistanceLoss
from torch_em.data.datasets import get_covid_if_loader
from torch_em.transform.label import PerObjectDistanceTransform
from torch_em.data import MinInstanceSampler

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model


def get_dataloaders(patch_shape, data_path):
    """This returns the livecell data loaders implemented in torch_em:
    https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/livecell.py
    It will automatically download the livecell data.

    Note: to replace this with another data loader you need to return a torch data loader
    that retuns `x, y` tensors, where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask instance segmentation format.
    I.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reseved for background, and the IDs must be consecutive
    """
    num_workers = 8 if torch.cuda.is_available() else 0

    label_transform = PerObjectDistanceTransform(
        distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True, min_size=25
    )
    raw_transform = sam_training.identity  # the current workflow avoids rescaling the inputs to [-1, 1]
    sampler = MinInstanceSampler()

    train_volumes = (None, 10)
    val_volumes = (10, 13)

    # let's estimate the total number of patches
    train_loader = get_covid_if_loader(
        path=data_path, patch_shape=patch_shape, batch_size=1, target="cells",
        download=True, sampler=sampler, sample_range=train_volumes
    )

    print(
        f"Found {len(train_loader)} samples for training.",
        "Hence, we will use {0} samples for training.".format(50 if len(train_loader) < 50 else len(train_loader))
    )

    # now, let's get the training and validation dataloaders
    
    train_loader = get_covid_if_loader(
        path=data_path, patch_shape=patch_shape, batch_size=1, target="cells", num_workers=num_workers, shuffle=True,
        raw_transform=raw_transform, sampler=sampler, label_transform=label_transform, label_dtype=torch.float32,
        sample_range=train_volumes, n_samples=50 if len(train_loader) < 50 else None,
    )

    val_loader = get_covid_if_loader(
        path=data_path, patch_shape=patch_shape, batch_size=1, target="cells", download=True, num_workers=num_workers,
        raw_transform=raw_transform, sampler=sampler, label_transform=label_transform, label_dtype=torch.float32,
        sample_range=val_volumes, n_samples=5,
    )

    return train_loader, val_loader


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params = params / 1e6
    return f"The number of trainable parameters for the provided model is {round(params, 2)}M"


def finetune_covid_if(args):
    """Code for finetuning SAM (using LoRA) on Covid IF

    Initial observations: There's no real memory advantage actually unless it's "truly" scaled up
    # vit_b
    # SAM: 93M (takes ~50GB)
    # SAM-LoRA: 4.2M (takes ~49GB)

    # vit_l
    # SAM: 312M (takes ~63GB)
    # SAM-LoRA: 4.4M (takes ~61GB)

    # vit_h
    # SAM: 641M (takes ~73GB)
    # SAM-LoRA: 4.7M (takes ~67GB)

    # Q: Would quantization lead to better results? (eg. QLoRA / DoRA)
    """
    # override this (below) if you have some more complex set-up and need to specify the exact gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training settings:
    model_type = args.model_type
    checkpoint_path = None  # override this to start training from a custom checkpoint  # the patch shape for training
    n_objects_per_batch = 5  # this is the number of objects per batch that will be sampled
    patch_shape = (512,512)
    freeze_parts = args.freeze  # override this to freeze different parts of the model
    rank = args.lora_rank  # the rank
    # get the trainable segment anything model
    model = sam_training.get_trainable_sam_model(
        model_type=model_type,
        device=device,
        checkpoint_path=checkpoint_path,
        freeze=freeze_parts,
        use_lora=args.use_lora,
        rank=rank,
    )
    model.to(device)

    # let's get the UNETR model for automatic instance segmentation pipeline
    unetr = UNETR(
        backbone="sam",
        encoder=model.sam.image_encoder,
        out_channels=3,
        use_sam_stats=True,
        final_activation="Sigmoid",
        use_skip_connection=False,
        resize_input=True,
    )

    # let's check the total number of trainable parameters
    print(count_parameters(model))

    # let's get the parameters for SAM and the decoder from UNETR
    joint_model_params = model.parameters()

    joint_model_params = [params for params in joint_model_params]  # sam parameters
    for name, params in unetr.named_parameters():  # unetr's decoder parameters
        if not name.startswith("encoder"):
            joint_model_params.append(params)

    optimizer = torch.optim.Adam(joint_model_params, lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=10)
    train_loader, val_loader = get_dataloaders(patch_shape=patch_shape, data_path=args.input_path)

    # this class creates all the training data for a batch (inputs, prompts and labels)
    convert_inputs = sam_training.ConvertToSamInputs(transform=model.transform, box_distortion_factor=0.025)
    name = (
        f"{args.model_type}/covid_if_"
        f"{f'lora_rank_{args.lora_rank}' if args.use_lora else 'sam'}"
    )


    trainer = sam_training.JointSamTrainer(
        name=name,
        save_root=args.save_root,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        device=device,
        lr_scheduler=scheduler,
        logger=sam_training.JointSamLogger,
        log_image_interval=100,
        mixed_precision=True,
        convert_inputs=convert_inputs,
        n_objects_per_batch=n_objects_per_batch,
        n_sub_iteration=8,
        compile_model=False,
        mask_prob=0.5,  # (optional) overwrite to provide the probability of using mask inputs while training
        unetr=unetr,
        instance_loss=DiceBasedDistanceLoss(mask_distances_in_bg=True),
        instance_metric=DiceBasedDistanceLoss(mask_distances_in_bg=True),
        early_stopping=10
    )
    trainer.fit(args.iterations)
    if args.export_path is not None:
        checkpoint_path = os.path.join(i
            "" if args.save_root is None else args.save_root, "checkpoints", args.name, "best.pt"
        )
        export_custom_sam_model(
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            save_path=args.export_path,
        )


def main():
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the CovidIF dataset.")
    parser.add_argument(
        "--input_path", "-i", default="/scratch/projects/nim00007/sam/data/covid_if/",
        help="The filepath to the CovidIF data. If the data does not exist yet it will be downloaded."
    )
    parser.add_argument(
        "--model_type", "-m", default="vit_b",
        help="The model type to use for fine-tuning. Either vit_h, vit_b or vit_l."
    )
    parser.add_argument(
        "--save_root", "-s", default=None,
        help="Where to save the checkpoint and logs. By default they will be saved where this script is run."
    )
    parser.add_argument(
        "--iterations", type=int, default=int(1e4),
        help="For how many iterations should the model be trained? By default 100k."
    )
    parser.add_argument(
        "--export_path", "-e",
        help="Where to export the finetuned model to. The exported model can be used in the annotation tools."
    )
    parser.add_argument(
        "--freeze", type=str, nargs="+", default=None,
        help="Which parts of the model to freeze for finetuning."
    )
    parser.add_argument(
        "--use_lora", action="store_true", help="Whether to use LoRA for finetuning."
    )
    parser.add_argument(
        "--lora_rank", type=int, default=4, help="Pass the rank for LoRA."
    )
    args = parser.parse_args()
    finetune_covid_if(args)


if __name__ == "__main__":
    main()
