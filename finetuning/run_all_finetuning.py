import os
import shutil
import subprocess
from datetime import datetime



def write_batch_script(out_path, _name, data, env_name, model_type, save_root, use_lora=False):
    "Writing scripts with different micro-sam finetunings."
    batch_script = f"""#!/bin/bash
#SBATCH -t 4-00:00:00
#SBATCH --mem 64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A nim00007
#SBATCH -c 16
#SBATCH --qos=96h
#SBATCH --constraint=80gb
#SBATCH --job-name={data}_{"lora" if use_lora else "full"}

source activate {env_name} \n"""

    # python script
    python_script = f"python {_name}.py "

    python_script += f"--data_name {data} "

    # save root folder
    python_script += f"-s {save_root} "

    # name of the model configuration
    python_script += f"-m {model_type} "

    # choice of the number of objects
    #python_script += f"--n_objects {N_OBJECTS[model_type[:5]]} "

    python_script += "--iterations 10000 "

    if use_lora:
        python_script += f"--use_lora "

    # let's add the python script to the bash script
    batch_script += python_script

    _op = out_path[:-3] + f"_{os.path.split(_name)[-1]}.sh"
    with open(_op, "w") as f:
        f.write(batch_script)

    cmd = ["sbatch", _op]
    subprocess.run(cmd)


def get_batch_script_names(tmp_folder):
    tmp_folder = os.path.expanduser(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)

    script_name = "micro-sam-finetuning"

    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")

    return batch_script


def submit_slurm(args):
    "Submit python script that needs gpus with given inputs on a slurm node."
    tmp_folder = "./gpu_jobs"

    data = {
        #"livecell": "~/micro-sam/finetuning/livecell/lora/train_livecell",
        #"covid_if": "~/micro-sam/finetuning/specialists/lora/covid_if_lora",
        #"mouse-embryo": "specialists/lora/mouse_embryo_lora",
        "orgasegment": "specialists/lora/orga_segment_lora",
        #"platy_cilia": "specialists/lora/platy_cilia_lora",
        #"mitolab_glycolytic_muscle": "specialists/lora/mytolab_lora",
    }
    if args.experiment_name is None:
        experiments = list(data.keys())
    else:
        assert args.experiment_name in list(data.keys()), \
            f"Choose from {list(data.keys())}"
        experiments = [args.experiment_name]


    for experiment in experiments:
        script_name = "~/micro-sam/finetuning/specialists/lora/train_lora"
        print(f"Running for {script_name}")

        if experiment in ["platy_cilia", "mitolab"]:
            model_checkpoint = "/scratch/projects/nim00007/sam/models/EM/generalist/v2/vit_b/best.pt"
            model_type = "vit_b_em_organelles"
        else:
            model_checkpoint = "/scratch/projects/nim00007/sam/models/LM/generalist/v2/vit_b/best.pt" 
            model_type = "vit_b_lm"

        # Full Finetuning
        write_batch_script(
            out_path=get_batch_script_names(tmp_folder),
            _name=script_name,
            data=experiment,
            env_name="sam",
            model_type=model_type,
            save_root=args.save_root
        )
        # Lora Finetuning
        write_batch_script(
            out_path=get_batch_script_names(tmp_folder),
            _name=script_name,
            data=experiment,
            env_name="sam",
            model_type=model_type,
            save_root=args.save_root,
            use_lora=True
        )


def main(args):
    try:
        shutil.rmtree("./gpu_jobs")
    except FileNotFoundError:
        pass

    submit_slurm(args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", type=str, default=None)
    parser.add_argument("-s", "--save_root", type=str, default="/scratch/usr/nimcarot/sam/experiments/SpecialistLoRA")
    parser.add_argument("-m", "--model_type", type=str, default="vit_b")
    args = parser.parse_args()
    main(args)
