import os
import json
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import streamlit as st
import random
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

# 设置 Streamlit 页面标题
st.set_page_config(page_title="ARC Data Viewer", layout="wide")
st.title("ARC Data Viewer")

DATA_ROOT = "data/arc-prize-2024/"

train_input_path = f'{DATA_ROOT}/arc-agi_training_challenges.json'
train_output_path = f'{DATA_ROOT}/arc-agi_training_solutions.json'

eval_input_path = f'{DATA_ROOT}/arc-agi_evaluation_challenges.json'
eval_output_path = f'{DATA_ROOT}/arc-agi_evaluation_solutions.json'

test_path = f'{DATA_ROOT}/arc-agi_test_challenges.json'
sample_path = f'{DATA_ROOT}/sample_submission.json'

eval_solutions_path = "outputs/solutions_eval.json"
test_solutions_path = "outputs/solutions.json"

# 加载数据
@st.cache_data
def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


PROMPT_TEMPLATE = PromptTemplate = """
You are given pairs of 2D matrices representing grids. In each matrix, 0 indicates the background, while identical non-zero numbers form specific zones and patterns.  
Your task is to identify the transformation rule that links each input matrix to its corresponding output matrix in the Examples. Then, apply this rule to generate an output matrix for the Test Input Matrix.

Specifically, you need to follow the steps below:
1. Focus on the size relationship between the input matrix and the output matrix in the Examples. There must be a clear dependency between the sizes of the matrices. Based on this, you should accurately determine the size of the output matrix from the Test Input Matrix.
2. Understand the transformation rule between the input matrix and the output matrix. These transformations are based on information from regions formed by identical numbers. This includes absolute positions and shapes of regions, relative positional relationships between regions, etc. You must have a clear definition and description of this transformation rule (but do not output it).
3. Based on the clearly understood transformation rule, strictly follow the output matrix size determined in the first step to generate the output matrix.
4. You only need to output the output matrix.

Examples:
{TRAIN}

Test Input Matrix:
{TEST}
"""

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
    num_rows = len(item['train']) + 1
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 4 * num_rows))
    
    # Plot test input matrix (only left side, no output)
    for j, test_item in enumerate(item['test']):
        plot_single_matrix(axes[0, 0], test_item['input'], title=f"Test Input {j + 1}")
        axes[0, 1].axis('off')  # Hide the empty plot on the right side

    # Plot train input and output matrices
    for i, train_item in enumerate(item['train']):
        plot_single_matrix(axes[i + 1, 0], train_item['input'], title=f"Train Input {i + 1}")
        plot_single_matrix(axes[i + 1, 1], train_item['output'], title=f"Train Output {i + 1}")

    plt.tight_layout()
    
    if if_draw:
        plt.show()
    else:
        plt.savefig(img_path)
        
    plt.close(fig)
    
    
def get_image(key, item):
    path = f"cache/{key}.png"
    if not os.path.isfile(path):
        os.makedirs("cache", exist_ok=True)
        visualize_item(item, img_path=path)
    
    return Image.open(path)


def plot_output_matrix(output_matrix, key, output_or_gt="output"):
    path = f"cache/{key}_{output_or_gt}.png"
    if not os.path.isfile(path):
        fig = plt.figure()  # 创建 figure 而非 subplots
        ax = fig.add_axes([0, 0, 1, 1])  # 创建单个轴 ax
        plot_single_matrix(ax, output_matrix, title="output")
        plt.savefig(path)
        plt.close(fig)
        
    return Image.open(path)    


# Select data option
data_option = st.selectbox("Select Data Set:", ("test", "val"))
# Load selected data
if data_option == "test":
    data = load_data(test_path)
    solutions = load_data(test_solutions_path)
    targets = None
elif data_option == "val":
    # val
    data = load_data(eval_input_path)
    solutions = load_data(eval_solutions_path)
    targets = load_data(eval_output_path)
else:
    # 暂时不支持 train
    pass

data = {k: data[k] for k in solutions}

# 显示数据总量
st.write(f"加载了 {len(data)} 个数据项")

# 数据浏览
item_keys = list(data.keys())
# selected_item = st.selectbox("选择数据项", item_keys)

# =====================================================================================================

# 将页面分为 4 列
# 初始化 session_state 中的 selected_item 和各个控件的初始状态
# Initialize or update session state variables when data_option changes
if 'previous_data_option' not in st.session_state or st.session_state['previous_data_option'] != data_option:
    # Update or initialize the session state variables with the new dataset
    st.session_state['selected_item'] = item_keys[0] if item_keys else None
    st.session_state['dropdown_select'] = st.session_state['selected_item']
    st.session_state['text_input'] = st.session_state['selected_item']
    st.session_state['current_i'] = item_keys.index(st.session_state['selected_item']) if item_keys else 0
    
    # Update the previous_data_option to the current data_option
    st.session_state['previous_data_option'] = data_option

# 定义更新函数，用于处理每个控件的值变化
def update_selected_item(source):
    # 当某个控件触发更新时，更新 selected_item，并同步其他控件的值
    if source == 'dropdown':
        st.session_state['selected_item'] = st.session_state['dropdown_select']
        st.session_state['text_input'] = st.session_state['selected_item']
    elif source == 'text':
        try:
            i = int(st.session_state['text_input'])
            input_str = item_keys[i]
        except:
            input_str = st.session_state['text_input']
        st.session_state['selected_item'] = input_str
        st.session_state['dropdown_select'] = st.session_state['selected_item']
    elif source == 'random':
        st.session_state['selected_item'] = random.choice(item_keys)

    # 更新 current_i 以反映 selected_item 在 item_keys 中的位置
    st.session_state['current_i'] = item_keys.index(st.session_state['selected_item'])
    print(st.session_state['current_i'])

# 布局
col1, col2, col3, col4 = st.columns(4)

# 第一列：选择数据项（下拉菜单）
with col1:
    st.selectbox("Select", item_keys, key="dropdown_select", 
                 on_change=lambda: update_selected_item('dropdown'))

# 第二列：输入框
with col2:
    st.text_input("Input", key="text_input", 
                  on_change=lambda: update_selected_item('text'))

# 第三列：随机选择按钮
with col3:
    if st.button("Random"):
        update_selected_item('random')

# 第四列：显示当前 i 并实现 +1 % len(item_keys)
with col4:
    st.write("Current i:", st.session_state['current_i'])
    if st.button("Next i"):
        # 将 current_i 增加 1 并取模
        st.session_state['current_i'] = (st.session_state['current_i'] + 1) % len(item_keys)
        # 更新 selected_item 以反映新的 current_i
        st.session_state['selected_item'] = item_keys[st.session_state['current_i']]
    
# =====================================================================================================

def format_matrix_string(matrix_string):
    # 添加矩阵外层格式，并处理每个子列表的换行和缩进
    formatted_string = "[\n    " + matrix_string[1:-1].replace("], [", "],\n    [") + "\n]"
    return formatted_string

# =========

selected_item = st.session_state['selected_item']
# update
if selected_item:
    
    print(selected_item)
    
    item = data[selected_item]
    input_img = get_image(selected_item, item)
    output_matrix_string = solutions[selected_item]

    if_correct = "No GT"
    if targets:
        gt_matrix = targets[selected_item][0]
        if_correct = json.dumps(gt_matrix) == item

    # 两列布局
    col1, col2, col3 = st.columns([2, 1, 2])

    # 左侧显示图像
    with col1:
        st.header("Input")
        if input_img:
            st.image(input_img, caption=f"{selected_item} Input", use_column_width=True)
        else:
            st.write("没有可用图像")

    # 右侧显示文本
    with col2:
        st.header("Output")
        
        try:
            output_matrix = json.loads(output_matrix_string)
            output_img = plot_output_matrix(output_matrix, key=selected_item, output_or_gt="output")
            st.image(output_img, caption=f"{selected_item} Output", use_column_width=True)
        except json.decoder.JSONDecodeError:
            st.text(output_matrix_string)
            
        if targets:
            gt_img = plot_output_matrix(gt_matrix, key=selected_item, output_or_gt="gt")
            st.image(gt_img, caption=f"{selected_item} GT", use_column_width=True)
        
    with col3:
        st.header("Text")
        # prompt_llm 显示区域
        prompt_llm = ""
        
        st.subheader("Prompt LLM")
        st.code(prompt_llm, language="text")
        
        # output 显示区域
        
        st.subheader("Output")
        st.header("If Correct")
        st.text(if_correct)
        
        st.code(output_matrix_string, language="text")
        st.code(format_matrix_string(output_matrix_string), language="text")
        
        if targets:
            st.code(format_matrix_string(json.dumps(gt_matrix)), language="text")

        st.text()
        
        # prompt_vlm
        prompt_vlm = ""
        
        st.subheader("Prompt VLM")
        st.code(prompt_vlm)
