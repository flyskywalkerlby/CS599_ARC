import os
import json
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import streamlit as st
import random

# 设置 Streamlit 页面标题
st.set_page_config(page_title="ARC VLM Data Viewer", layout="wide")
st.title("ARC VLM Data Viewer")

DATA_DIR = "data/arc_vlm"

# 加载数据
@st.cache_data
def load_data():
    data = defaultdict(dict)
    for filename in os.listdir(DATA_DIR):
        name, ext = os.path.splitext(filename)
        path = os.path.join(DATA_DIR, filename)
        if ext == ".json":
            with open(path, "r") as f:
                data[name]["text"] = json.load(f)
        elif ext == ".png":
            data[name]["img"] = Image.open(path)
    return dict(data)

PROMPT_VLM_TEMPLATE = """Train Matrices:
{Examples}

Test Matrix:
{Input}

You are given a set of matrices, including several training pairs and one test matrix. Each training pair consists of an input matrix and a corresponding output matrix. The goal is to identify the transformation rule that connects each input to its output, then apply this rule to generate the output for the test matrix.

Matrix Structure:

Training Pairs: Each training pair is displayed in a row with two sub-images:
Left Sub-image: The input matrix.
Right Sub-image: The output matrix.

Test Matrix: Shows only the input sub-image, with no output provided.

Each matrix contains numbers from 0 to 9, where identical non-zero numbers form clusters or shapes within the grid. The output matrix reflects a transformation based on patterns in the input matrix, influenced by factors such as size, shape, and position of these clusters.

Task:

Analyze the input-output pairs in the training data to identify a consistent transformation rule.
Apply this rule to generate the output matrix for the test input.
Output:
Produce the output matrix for the given test input matrix."""


PROMPT_LLM_TEMPLATE = """Training Matrices:
{Examples}

Test Matrix:
{Input}

Each matrix is a 2D grid where each entry is a single number from 0 to 9. Here, 0 represents the background, while identical consecutive non-zero numbers form distinct patterns or regions within the grid.

You are provided with multiple training matrix pairs and one test matrix. Each training pair contains an input matrix with a corresponding output matrix.

Task:
Identify the relationship and transformation between each input matrix and its output matrix. This relationship may involve aspects such as values, shapes, positions, and other pattern characteristics.
Possible transformations can include value changes, resizing, flipping, shifting, rotating, or more complex logical modifications.
Apply the identified transformation to the test input matrix to generate its output.

Output:
Generate and provide the output matrix for the test input matrix."""

# 加载数据
data = load_data()

# 显示数据总量
st.write(f"加载了 {len(data)} 个数据项")

# 数据浏览
item_keys = list(data.keys())
# selected_item = st.selectbox("选择数据项", item_keys)

# 将页面分为三列
col1, col2, col3 = st.columns(3)

# 第一列：选择数据项
with col1:
    selected_item = st.selectbox("选择数据项", item_keys)

# 第二列：输入框
with col2:
    selected_item = st.text_input("输入数据", "")

# 第三列：随机选择按钮
with col3:
    if st.button("随机选择"):
        selected_item = random.choice(item_keys)
        st.write(f"随机选择的数据项：{selected_item}")

if selected_item:
    item = data[selected_item]
    img = item.get("img", None)
    text_data = item.get("text", {})

    # 提取示例和输入的文本
    examples = text_data.get("examples", "")
    text_examples = "\n".join([json.dumps(example) for example in examples])
    
    input_data = text_data.get("input", "")
    text_input_data = json.dumps(input_data[0])
    
    output_data = text_data.get("output", "")
    output_data = output_data[0]
    text_output_data = "[\n" + "\n".join(["    " + str(row) + "," for row in output_data]) + "\n]"

    # 两列布局
    col1, col2 = st.columns([1, 2])

    # 左侧显示图像
    with col1:
        st.header("图像")
        if img:
            st.image(img, caption=f"{selected_item} 图像", use_column_width=True)
        else:
            st.write("没有可用图像")

    # 右侧显示文本
    with col2:
        st.header("文本")
        
        # prompt_llm 显示区域
        prompt_llm = PROMPT_LLM_TEMPLATE.format(
            Examples=text_examples,
            Input=text_input_data
        )
        
        st.subheader("Prompt LLM")
        st.code(prompt_llm, language="text")
        
        # output 显示区域
        st.subheader("Output")
        st.code(text_output_data, language="text")
        
        # prompt_vlm
        prompt_vlm = PROMPT_VLM_TEMPLATE.format(
            Examples=text_examples,
            Input=text_input_data
        )
        
        st.subheader("Prompt VLM")
        st.code(prompt_vlm)
