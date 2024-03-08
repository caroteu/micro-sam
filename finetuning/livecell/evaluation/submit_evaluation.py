import os
import re
import shutil
import subprocess
from glob import glob
from datetime import datetime
import argparse


def write_batch_script(env_name, out_path, inference_setup, checkpoint, model_type, experiment_folder, epoch, delay=None):
    """Writing scripts with different fold-trainings for micro-sam evaluation
    """
    batch_script = f"""#!/bin/bash
#SBATCH -c 8
#SBATCH --mem 128G
#SBATCH -t 2-00:00:00
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH --constraint=80gb
#SBATCH -A nim00007
#SBATCH --job-name={model_type}-{epoch}

source ~/.bashrc
mamba activate {env_name} \n"""
    
    if delay is not None:
        batch_script += f"sleep {delay} \n"

    with open(out_path, "w") as f:
        f.write(batch_script)

    for current_setup in inference_setup:

        # python script
        python_script = f"\npython ~/micro-sam/finetuning/livecell/evaluation/{current_setup}.py "

        # add the finetuned checkpoint
        python_script += f"-c {checkpoint} "

        # name of the model configuration
        python_script += f"-m {model_type} "

        # experiment folder
        python_script += f"-e {experiment_folder} "

        with open(out_path, "a") as f:
            f.write(python_script)

        # we run the first prompt for iterative once starting with point, and then starting with box (below)
        if current_setup == "iterative_prompting":
            python_script += "--box"

            with open(out_path, "a") as f:
                f.write(python_script)


def get_batch_script_names(tmp_folder, model_type, epoch):
    tmp_folder = os.path.expanduser(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)

    script_name = f"{model_type}_ep{epoch}_"

    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")

    return batch_script


def submit_slurm(epoch, model_type):
    """Submit python script that needs gpus with given inputs on a slurm node.
    """
    tmp_folder = f"./gpu_jobs"

    # parameters to run the inference scripts
    environment_name = "sam"
    experiment_set = "vanilla" if epoch == 0 else "specialist" # infer using specialists / generalists / vanilla models

    # let's set the experiment type - either using the specialists or generalists or just using vanilla model
    if experiment_set == "specialist":
        checkpoint = f"/scratch-emmy/usr/nimcarot/{model_type}/checkpoints/livecell_sam/epoch-{epoch}.pt"
        experiment_folder = f"/scratch/usr/nimcarot/sam/experiments/perf_over_epochs_2/epoch{epoch}/"

    elif experiment_set == "vanilla":
        checkpoint = None
        experiment_folder = f"/scratch/usr/nimcarot/sam/experiments/perf_over_epochs_2/epoch0/"

    else:
        raise ValueError("Choose from specialists / generalists / vanilla")

    experiment_folder += f"{model_type}/"

    # now let's run the experiments
    if experiment_set == "vanilla":
        all_setups = ["evaluate_amg", "iterative_prompting"]
    else:
        all_setups = ["evaluate_amg", "evaluate_instance_segmentation", "iterative_prompting"]

    batch_script_name = get_batch_script_names(tmp_folder, model_type, epoch)

    write_batch_script(
        env_name=environment_name,
        out_path=batch_script_name,
        inference_setup=all_setups,
        checkpoint=checkpoint,
        model_type=model_type,
        experiment_folder=experiment_folder,
        epoch=epoch,
        delay=None,
        )

    cmd = ["sbatch", batch_script_name]
    cmd_out = subprocess.run(cmd, capture_output=True, text=True)
    
    print(cmd_out.stdout if len(cmd_out.stdout) > 1 else cmd_out.stderr)


if __name__ == "__main__":
    try:
        shutil.rmtree("./gpu_jobs")
    except FileNotFoundError:
        pass

    for epoch in [2,3,4,5,6,7,8,9,10,20,30,40,50,60,62]:
        for model in ["vit_t", "vit_b", "vit_l", "vit_h"]:
            submit_slurm(epoch, model)
