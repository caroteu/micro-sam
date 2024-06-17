import os
import argparse

import torch

from torch_em.model import UNETR
from torch_em.loss import DiceBasedDistanceLoss
from torch_em.data.datasets import get_livecell_loader
from torch_em.transform.label import PerObjectDistanceTransform
from lion_pytorch import Lion

from get_loaders_for_lora import _fetch_loaders

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params = params / 1e6
    return f"The number of trainable parameters for the provided model is {round(params, 2)}M"


def finetune_livecell(args):
    """Code for finetuning SAM (using LoRA) on LIVECell

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
    checkpoint_path = None  # override this to start training from a custom checkpoint
    patch_shape = (520, 704)  # the patch shape for training
    n_objects_per_batch = 25  # this is the number of objects per batch that will be sampled
    freeze_parts = args.freeze  # override this to freeze different parts of the model

    # get the trainable segment anything model
    model = sam_training.get_trainable_sam_model(
        model_type=model_type,
        device=device,
        checkpoint_path=checkpoint_path,
        freeze=freeze_parts,
        get_lora=args.use_lora,
        rank=args.rank
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
    unetr.to(device)

    # let's check the total number of trainable parameters
    print(count_parameters(model))

    # let's get the parameters for SAM and the decoder from UNETR
    joint_model_params = model.parameters()

    joint_model_params = [params for params in joint_model_params]  # sam parameters
    for name, params in unetr.named_parameters():  # unetr's decoder parameters
        if not name.startswith("encoder"):
            joint_model_params.append(params)
            
    optimizer = torch.optim.AdamW(joint_model_params, lr=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=10)
    train_loader, val_loader = _fetch_loaders(args.data_name)

    # this class creates all the training data for a batch (inputs, prompts and labels)
    convert_inputs = sam_training.ConvertToSamInputs(transform=model.transform, box_distortion_factor=0.025)

    trainer = sam_training.JointSamTrainer(
        name=f"{args.data_name}_{'lora' if args.use_lora else 'full_ft'}",
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
        instance_metric=DiceBasedDistanceLoss(mask_distances_in_bg=True)
    )
    trainer.fit(args.iterations)
    if args.export_path is not None:
        checkpoint_path = os.path.join(
            "" if args.save_root is None else args.save_root, "checkpoints", "livecell_lora", "best.pt"
        )
        export_custom_sam_model(
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            save_path=args.export_path,
        )


def main():
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the LiveCELL dataset.")

    parser.add_argument(
        "--data_name", "-d", default="livecell",
        help="The name of the dataset to use for training."
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
        help="For how many iterations should the model be trained?."
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
        "--rank", type=int, default=4,
        help="The rank used in LoRA finetuning."
    )
    parser.add_argument(
        "--use_lora", action="store_true",
        help="Whether to use LoRA for finetuning."
    )
    args = parser.parse_args()
    finetune_livecell(args)


if __name__ == "__main__":
    main()
