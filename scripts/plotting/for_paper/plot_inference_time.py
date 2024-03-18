import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

EXPERIMENT_ROOT = "/scratch/usr/nimcarot/sam/experiments/benchmark/"


PALETTE = {"vit_t": "#089099", "vit_b": "#7CCBA2", "vit_l": "#FCDE9C", "vit_h": "#F0746E"}
MODELS = ['vit_t', 'vit_b', 'vit_l', 'vit_h']





def get_radar_plot(ax, dfs, model_name):
    
    plt.rcParams["hatch.linewidth"] = 1.5
    cat = dfs[0]['benchmark'].unique().tolist()
    cat = [*cat, cat[0]] #to close the radar, duplicate the first column
    n_points = len(cat)

    #As we have 5 categories the radar chart shoud have 5 radial axis
    # To find out the angle of each quadrant we divide 360/5 = 72 degrees
    #angles need to be converted to radian so we multiply by 2*pi and create the list of angles:

    plt.figure(figsize=(8, 8), facecolor="white")
    ax=plt.subplot(polar=True)
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(cat))
    for i,df in enumerate(dfs):
        norm = {}
        max_values = []
        for label in cat[:-1]:
            max_values.append(max([df_i.loc[df_i['benchmark'] == label]['runtimes'].item() for df_i in dfs]))
            norm[label] = df[df['benchmark'] == label]['runtimes'].item() / max_values[-1]

        group_norm = list(norm.values())
        group_norm += group_norm[:1]
        group = list(df['runtimes'])
        group += group[:1]

        err = df['error'] / np.array(max_values)
        err =np.array([*err, err[0]]) 

        ax.plot(label_loc, group_norm, 'o-', color=PALETTE[MODELS[i]], label=MODELS[i])

        for _x, _y, t in zip(label_loc, group_norm, group):
            t = f'{t:.2f}' if isinstance(t, float) else str(t)
            offset = 0.0
            ax.text(_x, _y-offset, t, va='top', ha='center', fontsize=10)


        ax.fill(label_loc, group_norm + err, facecolor=PALETTE[MODELS[i]], alpha=0.25)
        ax.fill(label_loc,  group_norm - err, facecolor="white", alpha=1)

    # Fix axis to go in the right order and start at 12 o'clock.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label.
    ax.set_thetagrids(np.degrees(label_loc), cat)
    ax.set_yticklabels([])


    for label, angle in zip(ax.get_xticklabels(), label_loc):
        if 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')
    # Add title.
    ax.set_title(f'{model_name}', y=1.08)

    # Add a legend as well.
    ax.legend()




def main():

    vit_t_gpu = pd.read_csv(f"{EXPERIMENT_ROOT}benchmark_vit_t_ft_1803.csv")    
    vit_b_gpu = pd.read_csv(f"{EXPERIMENT_ROOT}benchmark_vit_b_ft_1803.csv")    
    vit_l_gpu = pd.read_csv(f"{EXPERIMENT_ROOT}benchmark_vit_l_ft_1803.csv")    
    vit_h_gpu = pd.read_csv(f"{EXPERIMENT_ROOT}benchmark_vit_h_ft_1803.csv")    

    #vit_t_cpu = pd.read_csv(f"{EXPERIMENT_ROOT}benchmark_vit_t_ft_1003_cpu.csv")    
    #vit_b_cpu = pd.read_csv(f"{EXPERIMENT_ROOT}benchmark_vit_b_ft_1003_cpu.csv")    
    #vit_l_cpu = pd.read_csv(f"{EXPERIMENT_ROOT}benchmark_vit_l_ft_0403.csv")    
    #vit_h_cpu = pd.read_csv(f"{EXPERIMENT_ROOT}benchmark_vit_h_ft_0403.csv")    

    fig, ax = plt.subplots(1,2, figsize=(10,10))

    get_radar_plot(ax[0], [vit_t_gpu, vit_b_gpu, vit_l_gpu, vit_h_gpu], "GPU")
    #get_radar_plot(ax[0,1],[vit_t_cpu, vit_b_cpu, vit_l_cpu, vit_h_cpu], "CPU") 
    

    plt.show()
    print("Saving plot ...  ")
    plt.savefig("radar_plot.png")
    plt.close()


if __name__ == "__main__":
    main()