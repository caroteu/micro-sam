import os
from pathlib import Path
import pandas as pd
import seaborn as sns
from glob import glob
import matplotlib.pyplot as plt


EXPERIMENT_ROOT = "/scratch/usr/nimcarot/sam/experiments/"

# adding a fixed color palette to each experiments, for consistency in plotting the legends
PALETTE = {"vit_t": "#045275", "vit_b": "#089099", "vit_l": "#7CCBA2", "vit_h": "#FCDE9C"}
MODELS = ['vit_t', 'vit_b', 'vit_l', 'vit_h']


def gather_all_results():
    
    for rep in [2,3]:
        experiment = f"perf_over_epochs_{rep}"
        res_list_per_rep = []
        
        for experiment_dir in glob(os.path.join(EXPERIMENT_ROOT, experiment, "*")):
            if os.path.split(experiment_dir)[-1].startswith("epoch"):
                res_list_per_epoch = []
                epoch = os.path.split(experiment_dir)[-1].split('h')[-1]
                for i,result_dir in enumerate(glob(os.path.join(experiment_dir, '*', "results", '*'))):

                    if os.path.split(result_dir)[-1].startswith("grid_search_"):
                        continue
                    model = result_dir.split("/")[-3]
        
                
                    setting_name = Path(result_dir).stem
                    result = pd.read_csv(result_dir)
                    #print(i, result_dir)
                    #print(result)
                    if setting_name == "amg" or setting_name.startswith('instance'):
                        res_df = pd.DataFrame(
                            {
                                "name": setting_name,
                                "type": "none",
                                "rep": rep,
                                "model": model,
                                "epoch": epoch,
                                "result": result.iloc[0]["msa"]
                            }, index=[i]
                        )
                    else:
                        prompt_name = Path(result_dir).stem.split("_")[-1]
                        res_df = pd.concat(
                            [
                                pd.DataFrame(
                                    {"name": setting_name,
                                        "type": prompt_name,
                                        "rep": rep,
                                        "model": model,
                                        "epoch": epoch,
                                        "result": result.iloc[0]["msa"]},
                                        index = [i]
                                ),
                                pd.DataFrame(
                                    {"name": setting_name,
                                        "type": f"i_{prompt_name[0]}",
                                        "rep": rep,
                                        "model": model,
                                        "epoch": epoch,
                                        "result": result.iloc[-1]["msa"]},
                                        index=[i]
                                )
                            ], ignore_index=True
                        )
                    
                    res_list_per_epoch.append(res_df)
                   
        res_df_per_epoch = pd.concat(res_df_per_epoch)
        res_list_per_rep.append(res_df_per_epoch)
    res_df_per_rep = pd.concat(res_list_per_rep)

    return res_df_per_rep


def get_plots(ax, data, experiment_name):
    
    #data = data.groupby('epoch').agg('result':['mean', 'std'])
    sns.lineplot(data['epoch'], data['result'], ax=ax, hue=data['model'], palette=PALETTE, errorbar='sd', err_style='band')



def plot_perf_over_epochs():

    all_data = gather_all_results()
    print(all_data)
    fig, ax = plt.subplots(2,3, figsize=(20,15))

    amg = all_data[all_data["name"] == "amg"]
    ais = all_data[all_data["name"] == "instance_segmentation_with_decoder"]
    point = all_data[all_data["prompt"] == "point"]
    box = all_data[all_data["prompt"] == "box"]
    i_point = all_data[all_data["prompt"] == "i_point"]
    i_box = all_data[all_data["prompt"] == "i_box"]

    get_plots(ax[0,0], point, "point")
    get_plots(ax[0,1], box, "box")
    get_plots(ax[0,2], ais, "ais")
    get_plots(ax[1,0], amg, "amg")
    get_plots(ax[1,1], i_point, "iterative prompting (start with point)")
    get_plots(ax[1,2], i_box, "iterative prompting (start with box)")


    plt.show()


def main():
    plot_perf_over_epochs()

if __name__ == "__main__":
    main()