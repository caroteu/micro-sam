import os

from micro_sam.evaluation.evaluation import run_evaluation
from micro_sam.evaluation.inference import run_amg

from util import get_paths  # comment this and create a custom function with the same name to run amg on your data
from util import get_pred_paths, get_default_arguments, VANILLA_MODELS


def run_amg_inference(dataset_name, model_type, checkpoint, experiment_folder, use_lora=False, rank=None):
    val_image_paths, val_gt_paths = get_paths(dataset_name, split="val")
    test_image_paths, _ = get_paths(dataset_name, split="test")
    prediction_folder = run_amg(
        checkpoint,
        model_type,
        experiment_folder,
        val_image_paths,
        val_gt_paths,
        test_image_paths,
        use_lora=use_lora, 
        rank=rank
    )
    return prediction_folder


def eval_amg(dataset_name, prediction_folder, experiment_folder):
    print("Evaluating", prediction_folder)
    _, gt_paths = get_paths(dataset_name, split="test")
    pred_paths = get_pred_paths(prediction_folder)
    save_path = os.path.join(experiment_folder, "results", "amg.csv")
    res = run_evaluation(gt_paths, pred_paths, save_path=save_path)
    print(res)


def main():
    args = get_default_arguments()
    if args.checkpoint is None:
        ckpt = VANILLA_MODELS[args.model]
    else:
        ckpt = args.checkpoint

    prediction_folder = run_amg_inference(args.dataset, args.model, ckpt, args.experiment_folder, use_lora=args.use_lora, rank=args.lora_rank)
    eval_amg(args.dataset, prediction_folder, args.experiment_folder)


if __name__ == "__main__":
    main()
