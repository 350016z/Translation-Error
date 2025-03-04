import gradio as gr
import pandas as pd
import os
import time
import json
from pathlib import Path
from huggingface_hub import CommitScheduler, snapshot_download
from uuid import uuid4
from datasets import load_dataset
import shutil

DATASET_DIR = Path("json_dataset")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

scheduler = CommitScheduler(
    repo_id="350016z/TaiwanCOMET_dataset", 
    repo_type="dataset",
    folder_path=DATASET_DIR,
    path_in_repo="data"
)

# Loading dataset from HuggingFace -------------------------------------------------------------------------------------
def download_dataset_file(dataset_id, local_dir):
    # /home/user/.cache/huggingface/hub/datasets--350016z--Taiwanese_dataset/snapshots/22594253c63bd80e85b5255f948432014c37373a
    snapshot_path = snapshot_download(repo_id=dataset_id, repo_type="dataset")
    contents = os.listdir(snapshot_path)
    
    for file_name in contents:
        print("Checking file: ", file_name)
        if file_name.endswith(".csv"):
            source_file_path = os.path.join(snapshot_path, file_name)
            local_file_path = os.path.join(local_dir, file_name)
            
            shutil.copy(source_file_path, local_file_path)
            print(f"Copied {file_name} to {local_file_path}")
            
            # Check file permissions
            print(f"Permissions for {local_file_path}: {oct(os.stat(local_file_path).st_mode)}")

            time.sleep(1)
          
    return local_dir

DATASET_ID = "350016z/Taiwanese_dataset"
current_dir = os.getcwd()
download_dataset_file(DATASET_ID, current_dir)

csv_files = [f for f in os.listdir(current_dir) if f.endswith('.csv')]
if not csv_files:
    print("Error: No CSV files found in the current directory.")
    exit()

data_path = os.path.join(current_dir, 'test.csv') if 'test.csv' in csv_files else os.path.join(current_dir, csv_files[0])
print(f"Data path: {data_path}")

if not os.path.exists(data_path):
    print(f"Error: {data_path} does not exist. Please check the file path.")
    exit()


# Loading & Setting --------------------------------------------------------------------------------------------------
data = pd.read_csv(data_path, dtype={"id": "Int64"}) # 確保 id 為標準 Python int

current_index = 0
current_errors = []

annotations_file = DATASET_DIR / f"test_annotations-{uuid4()}.json"
# ---------------------------------------------------------------------------------------------------------------------

def get_all_ids():
    return [str(id) for id in data["id"].tolist()]
    
def get_current_text():
    global current_index, data
    source = data.loc[current_index, "source"]
    target = data.loc[current_index, "target"]
    return source, target

def save_to_json(entry: dict, json_file: Path):
    """
    將資料保存到指定的 JSON 檔案，並推送到 Hugging Face Dataset。
    """
    with scheduler.lock:
        with json_file.open("a") as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
        # scheduler.push_to_hub(commit_message=f"更新檔案 {json_file.name}")


def save_current(source, target, rater_selector, error_span, category, subcategory, severity, other):
    global current_index, data, current_errors
         
    system = data.loc[current_index, "system"]
    lp = data.loc[current_index, "lp"]
    doc = data.loc[current_index, "doc"]
    id = int(data.loc[current_index, "id"])
    reference = data.loc[current_index, "reference"]

    if subcategory:
        if subcategory == "Other":
            category_value = f"{category}/{other}"
        else:
            category_value = f"{category}/{subcategory}"
        

    if error_span and error_span in target:
        start = target.find(error_span)
        end = start + len(error_span)
        print(f"start: {start}, end: {end}")
    else:
        return "", "錯誤區間不存在於翻譯文本中，請檢查！"

    current_errors.append({
        "text": error_span,
        "severity": severity,
        "start": start,
        "end": end,
        "category": category_value,
    })

    # [error_span, status]
    return "", f"已記錄錯誤區間: {error_span}，範圍 {start}-{end}。"


def save_and_next(source, target, score, rater_selector):
    global current_index, data, annotations_file, current_errors

    system = data.loc[current_index, "system"]
    lp = data.loc[current_index, "lp"]
    doc = data.loc[current_index, "doc"]
    id = int(data.loc[current_index, "id"])
    reference = data.loc[current_index, "reference"]

    annotations_entry = {
        "system": system,
        "lp": lp,
        "doc": doc,
        "id": id,
        "rater": rater_selector,
        "src": source,
        "mt": target,
        "ref": reference,
        "esa_score": score,
        "esa_spans": current_errors,
    }
    save_to_json(annotations_entry, annotations_file)

    # 清空當前錯誤緩存
    current_errors = []

    current_index += 1
    if current_index >= len(data):
        return "已完成所有文本標記", "已完成所有文本標記", "", "", f"所有標記已完成並保存到 {annotations_file.name}！"

    next_source, next_target = get_current_text()
    return next_source, next_target, "", str(current_index), f"分數與錯誤已保存到 {annotations_file.name}，請繼續下一筆！"


def update_file_selection(selected_file):
    global data_path, data, current_index, annotations_file
    data_path = os.path.join(current_dir, selected_file)
    data = pd.read_csv(data_path)
    
    id_list = [str(id) for id in sorted(data["id"].unique())]  # 轉為字串，確保 Gradio Dropdown 兼容
    min_id = int(id_list[0])  # 取得最小的 ID
    
    current_index = data.index[data["id"] == int(min_id)].tolist()[0]  # DataFrame 的行索引（row index）；而非檔案中的id

    file_base_name = os.path.splitext(selected_file)[0]
    annotations_file = DATASET_DIR / f"{file_base_name}_annotations-{uuid4()}.json"

    # [source, target, error_span, index_selector, current_index_display, status]
    return get_current_text() + ("", gr.update(choices=id_list, value=str(min_id)), str(min_id), f"已加載檔案：{selected_file}")


def update_index_selection(selected_index):
    global current_index, data
    selected_index = int(selected_index)
    current_index = data.index[data["id"] == selected_index].tolist()[0]
    # [source, target, current_index_display, status]
    return get_current_text() + (str(selected_index), f"已跳轉至 id: {selected_index}")
        
categories = {
    "Accuracy": ["Mistranslation", "Addition", "Omission", "Other"],
    "Fluency": ["Grammar", "Spelling", "Punctuation", "Inconsistency", "Register", "Other"],
    "Terminology": ["Inappropriate", "Inconsistent", "Other"],
    "Style": ["Awkward", "Other"],
    "Locale": ["Currency format", "Time format", "Name format", "Date format", "Address format", "Other"],
}
severity_choices = ["Minor", "Major"]
rater = ['rater1', 'rater2','rater3', 'rater4', 'rater5', 'rater6', 'rater7']

def mark_as_correct():
    global current_errors

    current_errors.append({
        "text": "",
        "severity": "No-error",
        "start": 0,
        "end": 0,
        "category": "No-error"
    })
    return "", "標註為完全正確，無錯誤！"
    
def mark_as_too_many_errors():
    global current_errors

    current_errors.append({
        "text": "",
        "severity": "Major",
        "start": 0,
        "end": 0,
        "category": "Non-translation"
    })
    return "", "已標註為過多錯誤！"

DEMO_EXPLANATION = """
## 翻譯標記工具
### 使用規則 [更多細節](https://huggingface.co/spaces/350016z/TranslationError_Gradio/blob/main/README.md)
1. **開始作業**
    - 在「標註人員」選擇您的編號以識別。
    - 左側「原始文本」顯示原文，右側「翻譯文本」為機器翻譯結果，請檢查右側內容是否有錯誤。
2. **錯誤標註**
    - 發現翻譯錯誤時，將錯誤部分標註到「錯誤區間」欄位，錯誤需連接成最長可能區間，若中間有正確翻譯，需分段標註，避免連續標記。
    - 若有多處錯誤，可逐一標註並點擊「保存並繼續標記當前資料」後繼續修正。
    - 若錯誤超過五處，直接按下「過多錯誤」按鈕，再進行後續的評分。
    - 若無錯誤，直接按下「完全正確」按鈕，再進行後續的評分。
3. **評分**
    - 標記完所有錯誤區間以後，對每個翻譯文本的整體品質進行評分 (0-100分，0分最差，100分最好)。
        - 0：幾乎沒有保留原文意思，大部分資訊遺失。
        - 33：保留部分原文意思，但有明顯遺漏，敘述難以理解，文法可能很差。
        - 66：保留大部分原文意思，有一些文法錯誤或輕微不一致。
        - 100：原文意思和文法完全正確。 
        (即使選擇 **「完全正確」**，分數也不一定需要評100分)
"""

with gr.Blocks(css="""
    #correct_button {
        background-color: #4CAF50;
        color: white;
        font-size: 12px;
        padding: 5px 5px;
        border-radius: 5px;
        min-height: 0px;
    }
    #too_many_errors_button {
        background-color: #f44336;
        color: white;
        font-size: 12px;
        padding: 5px 5px;
        border-radius: 5px;
        min-height: 0px;
    }
""") as demo:
    gr.Markdown(DEMO_EXPLANATION)

    
    with gr.Tab("標記工具"):                
        with gr.Row():
            with gr.Column(scale=1):
                rater_selector = gr.Dropdown(label="標註人員", choices=rater, value="rater1")
                file_selector = gr.Dropdown(label="選擇檔案", choices=csv_files, value="test.csv")
                index_selector = gr.Dropdown(label="選擇索引", choices=get_all_ids())
                current_index_display = gr.Textbox(label="當前索引", value=str(current_index), interactive=False)
            with gr.Column(scale=8):
                source = gr.Textbox(label="原始文本", lines=15, interactive=False)
            with gr.Column(scale=8):
                target = gr.Textbox(label="翻譯文本", lines=15, interactive=False)

        with gr.Row(variant='panel', equal_height=True):
            with gr.Column(scale=3):
                error_span = gr.Textbox(label="錯誤區間 (💡可以直接複製「翻譯文本」欄位，並在此貼上)", lines=6, placeholder="請輸入翻譯中文本的錯誤區間 (如無錯誤則不需)")
            with gr.Column(scale=3):
                with gr.Row(equal_height=True):
                    category = gr.Dropdown(label="錯誤類別", choices=list(categories.keys()), value="Accuracy")
                    subcategory = gr.Dropdown(label="子類別", choices=categories["Accuracy"], value="Mistranslation")
                    other = gr.Textbox(label="其他子類別", placeholder="若無法歸類，請填寫其他")
                with gr.Row(equal_height=True):       
                    severity = gr.Radio(label="錯誤嚴重程度", choices=severity_choices, value="Minor")
                    save_current_button = gr.Button("保存並繼續標記當前資料")
            with gr.Column(scale=1):
                correct_button = gr.Button("✔ 完全正確", elem_id="correct_button")
                too_many_errors_button = gr.Button("✖ 過多錯誤", elem_id="too_many_errors_button")

        with gr.Row(variant='panel', equal_height=True):
            with gr.Column(scale=8):
                score = gr.Slider(label="翻譯評分", minimum=0, maximum=100, step=1, value=66)
            with gr.Column(scale=1):
                save_next_button = gr.Button("保存並顯示下一筆")

        status = gr.Textbox(label="當前狀態", lines=1, interactive=False)

        def update_subcategories(selected_category):
            subcategories = categories[selected_category]
            if subcategories:
                return gr.update(choices=subcategories, value=subcategories[0])
            else:
                return gr.update(choices=[], value=None)
        

        file_selector.change(update_file_selection, inputs=[file_selector], outputs=[source, target, error_span, index_selector, current_index_display, status])
        index_selector.change(update_index_selection, inputs=[index_selector], outputs=[source, target, current_index_display, status])
        category.change(update_subcategories, inputs=[category], outputs=[subcategory])
        
        correct_button.click(mark_as_correct, outputs=[error_span, status])
        too_many_errors_button.click(mark_as_too_many_errors, outputs=[error_span, status])

        save_current_button.click(save_current, inputs=[source, target, rater_selector, error_span, category, subcategory, severity, other], outputs=[error_span, status])
        save_next_button.click(save_and_next, inputs=[source, target, score, rater_selector], outputs=[source, target, error_span, current_index_display, status])

        original, translated = get_current_text()
        source.value = original
        target.value = translated
        
demo.launch()
