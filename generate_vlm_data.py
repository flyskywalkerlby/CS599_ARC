import json
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import colors
from tqdm import tqdm


DATA_ROOT = "data/arc-prize-2024/"

train_input_path = f'{DATA_ROOT}/arc-agi_training_challenges.json'
train_output_path = f'{DATA_ROOT}/arc-agi_training_solutions.json'

eval_input_path = f'{DATA_ROOT}/arc-agi_evaluation_challenges.json'
eval_output_path = f'{DATA_ROOT}/arc-agi_evaluation_solutions.json'

test_path = f'{DATA_ROOT}/arc-agi_test_challenges.json'
sample_path = f'{DATA_ROOT}/sample_submission.json'


with open(eval_input_path, "r") as f:
    input_data = json.load(f)

with open(eval_output_path, "r") as f:
    output_data = json.load(f)

data = {}
for i, k in enumerate(input_data):
    data[k] = dict(
        input=input_data[k],
        output=output_data[k]
    )
    if i == 0:
        print(k, data[k])

OUTPUT_DIR = "data/arc_vlm/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the custom colormap and normalization
cmap = colors.ListedColormap([
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
])
norm = colors.Normalize(vmin=0, vmax=9)


# Define the function to plot a single matrix with custom colors
def plot_single_matrix(ax, matrix, title=""):
    matrix = np.array(matrix)
    ax.imshow(matrix, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='white')


# Main function: visualize item with all subplots
def visualize_item(item, img_path=None, if_draw=False):
    num_rows = len(item['input']['train']) + 1
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 4 * num_rows))

    # Plot train input and output matrices
    for i, train_item in enumerate(item['input']['train']):
        plot_single_matrix(axes[i, 0], train_item['input'], title=f"Train Input {i + 1}")
        plot_single_matrix(axes[i, 1], train_item['output'], title=f"Train Output {i + 1}")

    # Plot test input matrix (only left side, no output)
    for j, test_item in enumerate(item['input']['test']):
        plot_single_matrix(axes[-1, 0], test_item['input'], title=f"Test Input {j + 1}")
        axes[-1, 1].axis('off')  # Hide the empty plot on the right side

    plt.tight_layout()
    
    if if_draw:
        plt.show()
    else:
        plt.savefig(img_path)
        
    plt.close(fig)


IF_DRAW = False

# 直接遍历并可视化 item 的所有矩阵内容
for i, (k, item) in enumerate(tqdm(data.items())):

    text_save_path = f"{OUTPUT_DIR}/{k}.json"
    img_save_path = f"{OUTPUT_DIR}/{k}.png"
    
    # try:
    #     with open(text_save_path, "r") as f:
    #         json.load(f)
    #     continue
    # except Exception as e:
    #     print(e)
    #     pass
    
    if os.path.isfile(text_save_path):
        continue

    ret = dict(
        examples=item["input"]["train"],
        input=item["input"]["test"],
        output=item["output"],
    )
    
    if IF_DRAW:
        print(k, ret)
    
    with open(text_save_path, "w") as f:
        json.dump(ret, f)
    
    visualize_item(item, img_path=img_save_path, if_draw=IF_DRAW)  
