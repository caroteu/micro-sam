import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


EXPERIMENT_ROOT = "/scratch/usr/nimcarot/sam/experiments/SpecialistLoRA/"
PROJECT_ROOT = "/scratch/projects/nim00007/sam/experiments/"


TOP_BAR_COLOR, BOTTOM_BAR_COLOR = "#F0746E", "#089099"

ALL_MODELS = {
    "vanilla": "Vanilla",
    "generalist": "Generalist",
    "specialist_lora": "LoRA Finetuning",
    "specialist_full_ft": "Full Finetuning"
}

MODEL_NAME_MAP = {
    "vit_b": "ViT Base",
    "vit_l": "ViT Large",
    "vit_h": "ViT Huge"
}

DATASETS = {
    "livecell": "LIVECell",
    "covid_if": "Covid IF",
    "mouse-embryo": "Mouse Embryo",
    "orgasegment": "OrgaSegment",
    "platy_cilia": "Platynereis (Cilia)",
    "mitolab_glycolytic_muscle": "MitoLab (Glycolytic Muscle)"
}
FIG_ASPECT = (30, 20)

plt.rcParams.update({'font.size': 30})


def gather_livecell_results(type, dataset="livecell", model="vit_b"):
    domain = "em" if dataset in ["mitolab_glycolytic_muscle", "platy_cilia"] else "lm"
    if type == "generalist":
        model = f"{model}_{domain}"
    result_paths = glob(
        os.path.join(
            EXPERIMENT_ROOT, type, domain, dataset, model, "results", "*"
        )
    )
    amg_score, ais_score, ib_score, ip_score = None, None, None, None
    for result_path in sorted(result_paths):
        if os.path.split(result_path)[-1].startswith("grid_search_"):
            continue
        print(result_path)
        res = pd.read_csv(result_path)
        setting_name = Path(result_path).stem
        if setting_name == "amg":
            amg_score = res.iloc[0]["msa"]
        elif setting_name.startswith("instance"):
            ais_score = res.iloc[0]["msa"]
        elif setting_name.endswith("box"):
            iterative_prompting_box = res["msa"]
            ib_score = [ibs for ibs in iterative_prompting_box]
        elif setting_name.endswith("point"):
            iterative_prompting_point = res["msa"]
            ip_score = [ips for ips in iterative_prompting_point]

    ip_score = pd.concat([
        pd.DataFrame(
            [{"iteration": idx, "name": "Point", "result": ip}]
        ) for idx, ip in enumerate(ip_score)
    ], ignore_index=True)

    ib_score = pd.concat([
        pd.DataFrame(
            [{"iteration": idx, "name": "Box", "result": ib}]
        ) for idx, ib in enumerate(ib_score)
    ], ignore_index=True)

    # let's get benchmark results
    cellpose_res = pd.read_csv(
        os.path.join(
            PROJECT_ROOT, "benchmarking", "cellpose", "livecell", "results", f"cellpose-livecell.csv"
        )
    )["msa"][0]

    return amg_score, ais_score, ib_score, ip_score, cellpose_res


def get_barplots(name, ax, ib_data, ip_data, amg, cellpose, ais=None, get_ylabel=True):
    sns.barplot(x="iteration", y="result", hue="name", data=ib_data, ax=ax, palette=[TOP_BAR_COLOR])
    if "error" in ib_data:
        ax.errorbar(
            x=ib_data['iteration'], y=ib_data['result'], yerr=ib_data['error'], fmt='none', c='black', capsize=20
        )

    sns.barplot(x="iteration", y="result", hue="name", data=ip_data, ax=ax, palette=[BOTTOM_BAR_COLOR])
    if "error" in ip_data:
        ax.errorbar(
            x=ip_data['iteration'], y=ip_data['result'], yerr=ip_data['error'], fmt='none', c='black', capsize=20
        )
    ax.set_xlabel("Iterations", labelpad=10, fontweight="bold")

    if get_ylabel:
        ax.set_ylabel("Mean Segmentation Accuracy", labelpad=10, fontweight="bold")
    else:
        ax.set_ylabel(None)

    ax.legend(title="Settings", bbox_to_anchor=(1, 1))
    ax.set_title(name, fontweight="bold")

    if amg is not None:
        ax.axhline(y=amg, label="AMG", color="#FCDE9C", lw=5)
    if ais is not None:
        ax.axhline(y=ais, label="AIS", color="#045275", lw=5)
    ax.axhline(y=cellpose, label="CellPose", color="#DC3977", lw=5)


def plot_for_livecell(dataset="livecell"):

    fig, ax = plt.subplots(2, 2, figsize=FIG_ASPECT, sharex=False, sharey=True)


    for i, experiment in enumerate(ALL_MODELS):
        print(experiment, dataset)
        if experiment == "default":
            amg, _, ib, ip, cellpose_res = gather_livecell_results(experiment, dataset)
            get_barplots(ALL_MODELS[experiment], ax[i//2][i%2], ib, ip, None, cellpose_res, get_ylabel=(i%2==0))
        else:
            amg, ais, ib, ip, cellpose_res = gather_livecell_results(experiment, dataset)
            get_barplots(ALL_MODELS[experiment], ax[i//2][i%2], ib, ip, None, cellpose_res, ais, get_ylabel=(i%2==0))

    # here, we remove the legends for each subplot, and get one common legend for all
    all_lines, all_labels = [], []
    for ax in fig.axes:
        lines, labels = ax.get_legend_handles_labels()
        for line, label in zip(lines, labels):
            if label not in all_labels:
                all_lines.append(line)
                all_labels.append(label)
        ax.get_legend().remove()

    fig.legend(all_lines, all_labels, loc="upper left", bbox_to_anchor=(0.1, 0.87))

    ax.set_yticks(np.linspace(0.1, 0.8, 8))

    plt.show()
    plt.tight_layout()
    fig.suptitle(DATASETS[dataset], fontsize=42, x=0.54, y=0.95, fontweight="bold")
    plt.subplots_adjust(right=0.95, left=0.1, top=0.87, bottom=0.1)
    _path = f"{dataset}_lora.svg" 
    plt.savefig(_path)
    plt.savefig(Path(_path).with_suffix(".pdf"))
    plt.close()



def main():
    for dataset in list(DATASETS.keys()):
        plot_for_livecell(dataset)

if __name__ == "__main__":
    main()

