import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FormatStrFormatter


ROOT = "/scratch/usr/nimcarot/sam/experiments/mito-orga/"

PALETTE = {
    "AIS": "#045275",
    "AMG": "#FCDE9C",
    "Point": "#7CCBA2",
    "Box": "#90477F",
}

plt.rcParams.update({"font.size": 30})


def _get_all_results(name, all_res_paths):
    all_res_list, all_box_res_list = [], []
    for i, res_path in enumerate(all_res_paths):
        res = pd.read_csv(res_path)
        res_name = Path(res_path).stem

        if res_name.startswith("grid_search"):
            continue

        if res_name.endswith("point"):
            res_name = "Point"
        elif res_name.endswith("box"):
            res_name = "Box"
        elif res_name.endswith("decoder"):
            res_name = "AIS"
        else:  # amg
            res_name = res_name.upper()

        res_df = pd.DataFrame(
            {"name": name, "type": res_name, "results": res.iloc[0]["msa"]}, index=[i]
        )
        if res_name == "Box":
            all_box_res_list.append(res_df)
        else:
            all_res_list.append(res_df)

    all_res_df = pd.concat(all_res_list, ignore_index=True)
    all_box_res_df = pd.concat(all_box_res_list, ignore_index=True)

    return all_res_df, all_box_res_df


def plot_all_experiments(dataset):
    all_experiment_paths = glob(os.path.join(ROOT, "*",dataset,"*", "results"), recursive=True)
    all_gen_box_res_list = []
    all_def_box_res_list = []
    all_gen_res_list = []
    all_def_res_list = []

    fig, ax = plt.subplots(2, 2, figsize=(30,30), sharey="row")
    for experiment_path in sorted(all_experiment_paths):
        rank = experiment_path.split('/')[-4]
        model = experiment_path.split('/')[-2]

        base_model = "Generalist" if len(model) > 5 else "Default"

        result_paths = glob(os.path.join(experiment_path, "*"))        
        all_res_df, all_box_res_df = _get_all_results(rank, result_paths)

        if base_model == "Default":
            all_def_res_list.append(all_res_df)
            all_def_box_res_list.append(all_box_res_df) 
        else:
            all_gen_res_list.append(all_res_df)
            all_gen_box_res_list.append(all_box_res_df)

    gen_res = pd.concat(all_gen_res_list)
    gen_box_res = pd.concat(all_gen_box_res_list)

    def_res = pd.concat(all_def_res_list)
    def_box_res = pd.concat(all_def_box_res_list)

    x_def_order = ['vanilla', 'lora_1', 'lora_2','lora_4', 'lora_8', 'lora_16', 'lora_32', 'lora_64', 'full_ft']
    x_gen_order = ['generalist', 'lora_1', 'lora_2','lora_4', 'lora_8', 'lora_16', 'lora_32', 'lora_64', 'full_ft']
   # Convert 'name' column to categorical with custom order
    gen_res['name'] = pd.Categorical(gen_res['name'], categories=x_gen_order, ordered=True)
    gen_box_res['name'] = pd.Categorical(gen_box_res['name'], categories=x_gen_order, ordered=True)
    def_res['name'] = pd.Categorical(def_res['name'], categories=x_def_order, ordered=True)
    def_box_res['name'] = pd.Categorical(def_box_res['name'], categories=x_def_order, ordered=True)
    
    # Sort DataFrames based on 'name' column
    gen_res_sorted = gen_res.sort_values(by='name')
    gen_box_res_sorted = gen_box_res.sort_values(by='name')
    def_res_sorted = def_res.sort_values(by='name')
    def_box_res_sorted = def_box_res.sort_values(by='name') 

    x_ticks_def = ['Vanilla', 'LoRA 1', 'LoRA 2', 'LoRA 4', 'LoRA 8', 'LoRA 16', 'LoRA 32', 'LoRA 64', 'Full FT']
    x_ticks_gen = ['Generalist', 'LoRA 1', 'LoRA 2', 'LoRA 4', 'LoRA 8', 'LoRA 16', 'LoRA 32', 'LoRA 64', 'Full FT']

    sns.lineplot(
        x="name", y="results", hue="type", data=def_box_res_sorted,
        ax=ax[0, 0], palette=PALETTE, hue_order=PALETTE.keys(),
        marker="o", markersize=15, linewidth=5
    )
    ax[0, 0].set_title("Default", fontweight="bold")
    ax[0, 0].set(xlabel=None, ylabel=None)
    ax[0, 0].set_yticks(np.linspace(0.6, 0.8, 3))
    ax[0, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    sns.lineplot(
        x="name", y="results", hue="type", data=def_res_sorted,
        ax=ax[1, 0], palette=PALETTE, hue_order=PALETTE.keys(),
        marker="o", markersize=15, linewidth=5
    )
    # ax[1, idx].set_title(_title, fontweight="bold")
    ax[1, 0].set(xlabel=None, ylabel=None)
    ax[1, 0].set_yticks(np.linspace(0.2, 0.6, 5))
    ax[1, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
 
    sns.lineplot(
        x="name", y="results", hue="type", data=gen_box_res_sorted,
        ax=ax[0, 1], palette=PALETTE, hue_order=PALETTE.keys(),
        marker="o", markersize=15, linewidth=5
    )
    ax[0, 1].set_title("Generalist", fontweight="bold")
    ax[0, 1].set(xlabel=None, ylabel=None)
    ax[0, 1].set_yticks(np.linspace(0.6, 0.8, 3))
    ax[0, 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    sns.lineplot(
        x="name", y="results", hue="type", data=gen_res_sorted,
        ax=ax[1, 1], palette=PALETTE, hue_order=PALETTE.keys(),
        marker="o", markersize=15, linewidth=5
    )
    # ax[1, idx].set_title(_title, fontweight="bold")
    ax[1, 1].set(xlabel=None, ylabel=None)
    ax[1, 1].set_yticks(np.linspace(0.2, 0.6, 5))
    ax[1, 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


    all_lines, all_labels = [], []
    for ax in fig.axes:
        lines, labels = ax.get_legend_handles_labels()
        for line, label in zip(lines, labels):
            if label not in all_labels:
                all_lines.append(line)
                all_labels.append(label)
        ax.get_legend().remove()

    custom_handles = []
    for color in PALETTE.values():
        line = mlines.Line2D([], [], color=color, markersize=15, marker='o', linestyle='-', linewidth=5)
        custom_handles.append(line)

    fig.legend(custom_handles, PALETTE.keys(), loc="lower center", ncols=4, bbox_to_anchor=(0.5, 0))

    def format_y_tick_label(value, pos):
        return "{:.2f}".format(value)
    

    for ax in fig.axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    plt.suptitle(dataset)

    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_tick_label))

    plt.text(x=-11.8, y=0.55, s="Mean Segmentation Accuracy", rotation=90, fontweight="bold")
    plt.text(x=-2.55, y=-0.05, s="Training Setting", fontweight="bold")

    plt.subplots_adjust(wspace=0.1, hspace=0.25, bottom=0.15, top=0.88)


    save_path = f"/user/teuber5/u12094/micro-sam/results/lora_datasets/{dataset}_extended.png"
    plt.savefig(save_path)
    plt.savefig(Path(save_path).with_suffix(".svg"))
    plt.savefig(Path(save_path).with_suffix(".pdf"))
    plt.close()

    full_gen_df = pd.concat([gen_res, gen_box_res])
    full_gen_df = full_gen_df.pivot(index='name', columns='type', values='results')
    print(full_gen_df.to_markdown())

    full_def_df = pd.concat([def_res, def_box_res])
    full_def_df = full_def_df.pivot(index='name', columns='type', values='results')
    print(full_def_df.to_markdown())

def main():
    plot_all_experiments("orgasegment")
    #plot_all_experiments("mitolab_glycolytic_muscle")


if __name__ == "__main__":
    main()