import os
from glob import glob
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

PROJECT_ROOT = "/scratch/projects/nim00007/sam/experiments"
MODEL_NAMES = {"vit_t": "ViT-T", "vit_b": "ViT-B", "vit_l": "ViT-L", "vit_h": "ViT-H"}
EXPERIMENT_ROOT = "/scratch/usr/nimcarot/sam/experiments/livecell"

def gather_livecell_results(model_type, experiment_name, benchmark_choice):
    result_paths = glob(
        os.path.join(
            EXPERIMENT_ROOT, experiment_name, model_type, "results", "*"
        )
    )
    ais_score = None
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
            [{"iteration": idx + 1, "name": "point", "result": ip}]
        ) for idx, ip in enumerate(ip_score)
    ], ignore_index=True)

    ib_score = pd.concat([
        pd.DataFrame(
            [{"iteration": idx + 1, "name": "box", "result": ib}]
        ) for idx, ib in enumerate(ib_score)
    ], ignore_index=True)

    # let's get benchmark results
    cellpose_res = pd.read_csv(
        os.path.join(
            PROJECT_ROOT, "benchmarking", "cellpose", "livecell", "results", f"cellpose-{benchmark_choice}.csv"
        )
    )["msa"][0]

    return amg_score, ais_score, ib_score, ip_score, cellpose_res


def get_barplots(name, ax, ib_data, ip_data, amg, cellpose, model, ais=None):
    data = pd.concat([ib_data, ip_data], ignore_index=True)
    sns.barplot(x="iteration", y="result", hue="name", data=data, ax=ax, palette=["#7CCBA2", '#089099'])
    #äall_containers = ax.containers[-1]
    #for k in range(len(all_containers)):
    #    ax.patches[k].set_hatch('///')
    #    ax.patches[k].set_edgecolor('k')

    #sns.barplot(x="iteration", y="result", hue="name", data=ip_data, ax=ax, palette=["#089099"])
    ax.set(xlabel=None, ylabel=None)
    ax.legend(title="Settings", bbox_to_anchor=(1, 1))
    ax.title.set_text(f"{name} {model}")

    ax.axhline(y=amg, label="amg", color="#7c1D6F")
    if ais is not None:
        ax.axhline(y=ais, label="ais", color="darkorange")
    ax.axhline(y=cellpose, label="cellpose", color="#E31A1C")



def plot_for_livecell(benchmark_choice):
    fig, ax = plt.subplots(4, 2, figsize=(20, 30), sharex="col", sharey="row")
    for i, m in enumerate(["vit_t", "vit_b", "vit_l", "vit_h"]):
        amg_vanilla, _, ib_vanilla, ip_vanilla, cellpose_res = gather_livecell_results(m, "vanilla", benchmark_choice)
        get_barplots("Default SAM", ax[i][0], ib_vanilla, ip_vanilla, amg_vanilla, cellpose_res, MODEL_NAMES[m])

        (amg_specialist, ais_specialist,
         ib_specialist, ip_specialist, cellpose_res) = gather_livecell_results(m, "specialist", benchmark_choice)
        get_barplots("Finetuned SAM", ax[i][1], ib_specialist, ip_specialist, amg_specialist, cellpose_res, MODEL_NAMES[m], ais_specialist)

    # here, we remove the legends for each subplot, and get one common legend for all
    all_lines, all_labels = [], []
    for ax in fig.axes:
        lines, labels = ax.get_legend_handles_labels()
        for line, label in zip(lines, labels):
            if label not in all_labels:
                all_lines.append(line)
                all_labels.append(label)
        ax.get_legend().remove()

    fig.legend(all_lines, all_labels, bbox_to_anchor=(0.11, 0.98), fontsize=13)

    fig.text(0.5, 0.01, 'Iterative Prompting', ha='center', fontdict={"size": 23})
    fig.text(0.01, 0.5, 'Segmentation Quality', va='center', rotation='vertical', fontdict={"size": 23})

    plt.show()
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, right=0.97, left=0.05, bottom=0.03)
    fig.suptitle("LiveCELL", fontsize=26)
    plt.savefig("livecell.pdf")
    plt.close()


def main():
    plot_for_livecell(benchmark_choice="livecell")


if __name__ == "__main__":
    main()
