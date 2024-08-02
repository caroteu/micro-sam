import re
import os
import shutil
import subprocess
from glob import glob
from pathlib import Path
from datetime import datetime


ALL_DATASETS = {'livecell':'lm', 'covid_if':'lm', 'orgasegment':'lm', 'mouse-embryo':'lm', 'mitolab/glycolytic_muscle':'em_organelles', 'platynereis/cilia':'em_organelles'}

ALL_SCRIPTS = [
    "precompute_embeddings", "evaluate_amg", "iterative_prompting", "evaluate_instance_segmentation"
]


def write_batch_script(
    env_name, out_path, inference_setup, checkpoint, model_type,
    experiment_folder, dataset_name, delay=None, use_masks=False
):
    "Writing scripts with different fold-trainings for micro-sam evaluation"
    batch_script = f"""#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 4-00:00:00
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A nim00007
#SBATCH --constraint=80gb
#SBATCH --qos=96h
#SBATCH --job-name={inference_setup}

source ~/.bashrc
mamba activate {env_name} \n"""

    if delay is not None:
        batch_script += f"sleep {delay} \n"

    # python script
    inference_script_path = os.path.join(Path(__file__).parent, f"{inference_setup}.py")
    python_script = f"python {inference_script_path} "

    _op = out_path[:-3] + f"_{inference_setup}.sh"

    if checkpoint is not None:# add the finetuned checkpoint
        python_script += f"-c {checkpoint} "

    # name of the model configuration
    python_script += f"-m {model_type} "

    # experiment folder
    python_script += f"-e {experiment_folder} "

    # IMPORTANT: choice of the dataset
    python_script += f"-d {dataset_name} "

    # use logits for iterative prompting
    if inference_setup == "iterative_prompting" and use_masks:
        python_script += "--use_masks "

    # let's add the python script to the bash script
    batch_script += python_script

    print(batch_script)
    with open(_op, "w") as f:
        f.write(batch_script)

    # we run the first prompt for iterative once starting with point, and then starting with box (below)
    if inference_setup == "iterative_prompting":
        batch_script += "--box "

        new_path = out_path[:-3] + f"_{inference_setup}_box.sh"
        with open(new_path, "w") as f:
            f.write(batch_script)


def get_batch_script_names(tmp_folder):
    tmp_folder = os.path.expanduser(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)

    script_name = "micro-sam-inference"

    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")

    return batch_script


def get_checkpoint_path(experiment_set, dataset_name, model_type, region):
    # let's set the experiment type - either using the generalist or just using vanilla model
    if experiment_set == "generalist":
        checkpoint = None
        # set new model_type to vit_b_em_organelles or vit_b_lm
        model_type = f"{model_type}_{region}"
        """
        if region == "organelles":
            checkpoint += "mito_nuc_em_generalist_sam/best.pt"
        elif region == "boundaries":
            checkpoint += "boundaries_em_generalist_sam/best.pt"
        elif region == "lm":
            checkpoint += "lm_generalist_sam/best.pt"
        else:
            raise ValueError("Choose `region` from lm / organelles / boundaries")
        """

    elif experiment_set == "specialist" or experiment_set == "specialist_lora":
        finetuning_type = "_lora" if experiment_set == "specialist_lora" else ""
        _split = dataset_name.split("/")
        if len(_split) > 1:
            # it's the case for plantseg/root, we catch it and convert it to the expected format
            dataset_name = f"{_split[0]}_{_split[1]}"

        # HACK:
        if dataset_name.startswith("neurips-cell-seg"):
            dataset_name = "neurips_cellseg"
        if dataset_name.startswith("asem"):
            dataset_name = "asem_er"
        if dataset_name.startswith("tissuenet"):
            dataset_name = "tissuenet"
        if dataset_name.startswith("platynereis"):
            dataset_name = "platy_cilia"
        if dataset_name.startswith("mitolab"):
            dataset_name = "mitolab_glycolytic_muscle"

        checkpoint = f"/scratch/usr/nimcarot/sam/experiments/covid_if/checkpoints/vit_b_lm/{dataset_name}_sam{finetuning_type}/best.pt"

    elif experiment_set == "vanilla" or experiment_set == "generalist":
        checkpoint = None

    else:
        raise ValueError("Choose from generalist / vanilla")

    if checkpoint is not None:
        assert os.path.exists(checkpoint), checkpoint

    return checkpoint


def submit_slurm(dataset_name, experiment_set, region, specific_experiment, args):
    "Submit python script that needs gpus with given inputs on a slurm node."
    tmp_folder = "./gpu_jobs"

    model_type = f"{args.model_type}_{region}" if experiment_set == "generalist" else args.model_type

    make_delay = "10s"  # wait for precomputing the embeddings and later run inference scripts

    if args.checkpoint_path is None:
        checkpoint = get_checkpoint_path(experiment_set, dataset_name, model_type, region)
    else:
        checkpoint = args.checkpoint_path

    if args.experiment_path is None:
        modality = region if region == "lm" else "em"
        experiment_folder = "/scratch/usr/nimcarot/sam/experiments/lora/"
        if experiment_set == "specialist":
            checkpoint_name = checkpoint.split("/")[-5:-2]
            experiment_folder += f"{experiment_set}/{dataset_name}/{checkpoint_name}/{model_type}/"
        experiment_folder += f"{experiment_set}/{modality}/{dataset_name}/{model_type}/"
    else:
        experiment_folder = args.experiment_path

    # now let's run the experiments
    if specific_experiment is None:
        if experiment_set == "vanilla":
            all_setups = ALL_SCRIPTS[:-1]
        else:
            all_setups = ALL_SCRIPTS
    else:
        assert specific_experiment in ALL_SCRIPTS
        all_setups = [specific_experiment]

    # env name
    if model_type == "vit_t":
        env_name = "mobilesam"
    else:
        env_name = "sam"

    for current_setup in all_setups:
        print("Write batch script for", current_setup, checkpoint, model_type, dataset_name)
        write_batch_script(
            env_name=env_name,
            out_path=get_batch_script_names(tmp_folder),
            inference_setup=current_setup,
            checkpoint=checkpoint,
            model_type=model_type,
            experiment_folder=experiment_folder,
            dataset_name=dataset_name,
            delay=None if current_setup == "precompute_embeddings" else make_delay,
            use_masks=args.use_masks
            )

    # the logic below automates the process of first running the precomputation of embeddings, and only then inference.
    job_id = []
    for i, my_script in enumerate(sorted(glob(tmp_folder + "/*"))):
        cmd = ["sbatch", my_script]

        if i > 0:
            cmd.insert(1, f"--dependency=afterany:{job_id[0]}")

        cmd_out = subprocess.run(cmd, capture_output=True, text=True)
        print(cmd_out.stdout if len(cmd_out.stdout) > 1 else cmd_out.stderr)

        if i == 0:
            job_id.append(re.findall(r'\d+', cmd_out.stdout)[0])


def main(args):
    try:
        shutil.rmtree("./gpu_jobs")
    except FileNotFoundError:
        pass

    #for dataset_name in list(ALL_DATASETS.keys())[:-2]:
    dataset_name = "covid_if"
    for experiment_set in ["specialist"]:
        roi = ALL_DATASETS[dataset_name]
        try:
            shutil.rmtree("./gpu_jobs")
        except FileNotFoundError: 
            pass
        submit_slurm(dataset_name, experiment_set, roi, None, args)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # the parameters to use the default models
    parser.add_argument("-m", "--model_type", type=str, required=True)
    parser.add_argument("-e", "--experiment_set", type=str)
    # optional argument to specify for the experiment root folder automatically
    parser.add_argument("-r", "--roi", type=str)
    parser.add_argument("--use_masks", action="store_true")

    # overwrite the checkpoint path and experiment root to use this flexibly
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--experiment_path", type=str, default=None)

    # ask for a specific experiment
    parser.add_argument("-s", "--specific_experiment", type=str, default=None)

    args = parser.parse_args()
    main(args)

