import gradio as gr
import pandas as pd
import os

data_path = "evaluation_dataset/input/test.csv"
annotations_path = "evaluation_dataset/output/test_annotations.csv"  
score_path = "evaluation_dataset/output/test_score.csv" 

current_dir = os.path.dirname(os.path.abspath(data_path))
csv_files = [f for f in os.listdir(current_dir) if f.endswith('.csv')]

data = pd.read_csv(data_path)
current_index = 0

# output
output_dir = os.path.join(current_dir, "output")
if not os.path.exists(output_dir): 
    os.makedirs(output_dir)

annotations_columns = ["source", "target", "rater", "error_span", "correction_text", "category", "severity", "other"]
score_columns = ["source", "target", "rater", "score"]

def get_current_text():
    global current_index, data
    if current_index < len(data):
        source = data.loc[current_index, "source"]
        target = data.loc[current_index, "target"]
        return source, target
    else:
        return "已完成所有文本標記", "已完成所有文本標記"
    
def save_current(source, target, rater_selector, error_span, correction_text, category, subcategory, severity, other):
    global annotations_path
    if category != "Non-translation":
        category_value = f"{category}/{subcategory}"
    else: # Non-translation
        category_value = category

    new_entry = {
        "source": source,
        "target": target,
        "rater": rater_selector,
        "error_span": error_span,
        "correction_text": correction_text,
        "category": category_value,
        "severity": severity,
        "other": other if other else ""
    }
    file_exists = os.path.isfile(annotations_path) 
    pd.DataFrame([new_entry]).to_csv(annotations_path, mode="a", header=not file_exists, index=False)
    return source, target, "", "", f"當前資料已保存，請繼續標記！"

def save_and_next(source, target, rater_selector, error_span, correction_text, category, subcategory, severity, other, score):
    global current_index, data, annotations_path, score_path
    new_entry_score = {
        "source": source,
        "target": target,
        "rater": rater_selector,
        "score": score,
    }
    file_exists = os.path.isfile(score_path)
    pd.DataFrame([new_entry_score]).to_csv(score_path, mode="a", header=not file_exists, index=False)

    current_index += 1
    if current_index < len(data):
        next_source, next_target = get_current_text()
        return next_source, next_target, "", "", f"標記已保存，請繼續下一筆！"
    else:
        return "已完成所有文本標記", "已完成所有文本標記", "", "", f"所有標記已完成並保存！"

def update_file_selection(selected_file):
    global data_path, data, current_index, annotations_path, score_path
    data_path = os.path.join(current_dir, selected_file)
    data = pd.read_csv(data_path)
    current_index = 0

    file_base_name = os.path.splitext(selected_file)[0]
    annotations_path = f"evaluation_dataset/output/{file_base_name}_annotations.csv"
    score_path = f"evaluation_dataset/output/{file_base_name}_score.csv"

    return get_current_text() + ("", "", f"已加載檔案：{selected_file}")

categories = {
    "No-error": [],
    "Accuracy": ["Mistranslation", "Addition", "Omission", "Other"],
    "Fluency": ["Grammar", "Spelling", "Punctuation", "Inconsistency", "Register", "Other"],
    "Terminology": ["Inappropriate", "Inconsistent", "Other"],
    "Style": ["Awkward", "Other"],
    "Locale": ["Currency format", "Time format", "Name format", "Date format", "Address format", "Other"],
    "Non-translation": []
}
rater = ['rater1', 'rater2','rater3', 'rater4', 'rater5', 'rater6', 'rater7']

with gr.Blocks(theme=gr.themes.Default()) as demo:
    with gr.Tab("標記工具"):
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(
                    """
                    # 翻譯標記工具
                    - 作業開始前，在「標註人員」選擇您的編號，作為日後識別。
                    - 左邊顯示原文，右邊為機器翻譯結果，請檢查右側內容是否有錯誤。
                    - 發現錯誤時，將錯誤部分輸入或複製到「錯誤區間」欄位，並在「正確內容」輸入正確翻譯。
                    - 若有多處錯誤，可逐一修正並點擊「保存並繼續標記當前資料」後繼續修正。
                    - 若錯誤超過五處，嚴重程度選擇「Major」及錯誤類別選擇「Non-translation」，無需修正。
                    - 當前該筆文本的錯誤均修正完畢後，請對您修正前的翻譯進行評分，再進入下一筆資料。
                    """
                    , elem_id="title"
                )
            with gr.Column(scale=1):
                rater_selector = gr.Dropdown(
                    label="標註人員",
                    choices=rater,
                    value="rater1"
                )
                file_selector = gr.Dropdown(
                    label="選擇檔案",
                    choices=csv_files,
                    value="test.csv"
                )
                
        with gr.Row():
            source = gr.Textbox(
                label="原始文本",
                lines=5,
                interactive=False
            )
            target = gr.Textbox(
                label="翻譯文本（請手動複製錯誤部分）",
                lines=5,
                interactive=False
            )
        with gr.Row():
            error_span = gr.Textbox(
                label="錯誤區間",
                lines=2,
                placeholder="請複製並貼上翻譯中的錯誤文本 (如無錯誤則不需)"
            )
            correction_text = gr.Textbox(
                label="正確內容",
                lines=2,
                placeholder="輸入修正後的正確內容 (如無錯誤則不需)"
            )

        # ['default', 'panel', 'compact']
        with gr.Row(variant='panel', equal_height=True):
            severity = gr.Radio(
                label="錯誤嚴重程度",
                choices=["No-error", "Minor", "Major"],
                value="No-error"
            )
            category = gr.Dropdown(label="錯誤類別", choices=list(categories.keys()), value="No-error")
            subcategory = gr.Dropdown(label="子類別", choices=categories["No-error"], value=None)
            other = gr.Textbox(label="其他子類別", placeholder="若無法歸類，請填寫其他")
            save_current_button =  gr.Button(value="保存並繼續標記當前資料")

        def update_subcategories(selected_category):
            subcategories = categories[selected_category]
            if subcategories:
                return gr.update(choices=subcategories, value=subcategories[0])
            else:
                return gr.update(choices=[], value=None)

        category.change(update_subcategories, inputs=[category], outputs=[subcategory])

        with gr.Row(variant='panel', equal_height=True):
            score = gr.Slider(
                label="翻譯評分 (0: 極差, 100: 完美)",
                minimum=0,
                maximum=100,
                step=1,
                value=50
            )
            save_next_button = gr.Button("保存並顯示下一筆")

        status = gr.Textbox(label="當前狀態", lines=1, interactive=False)

        file_selector.change(
            update_file_selection,
            inputs=[file_selector],
            outputs=[source, target, error_span, correction_text, status]
        )

        save_current_button.click(
            save_current,
            inputs=[source, target, rater_selector, error_span, correction_text, category, subcategory, severity, other],
            outputs=[source, target, error_span, correction_text, status]
        )

        save_next_button.click(
            save_and_next,
            inputs=[source, target, rater_selector, error_span, correction_text, category, subcategory, severity, other, score],
            outputs=[source, target, error_span, correction_text, status]
        )

        original, translated = get_current_text()
        source.value = original
        target.value = translated
    
    with gr.Tab("下載資料"):
        output_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]

        def get_output_files():
            return [f for f in os.listdir(output_dir) if f.endswith('.csv')]

        def refresh_output_files():
            refreshed_files = get_output_files()
            return gr.update(choices=refreshed_files, value=refreshed_files[0] if refreshed_files else None), "已刷新輸出檔案列表。"

        with gr.Row(equal_height=True):
            output_file_selector = gr.Dropdown(
                label="選擇輸出檔案",
                choices=get_output_files(),
                value=get_output_files()[0] if output_files else None
            )
            refresh_button = gr.Button("Refresh")
            download_button = gr.DownloadButton("Download")
            
        download_status = gr.Textbox(label="下載狀態", lines=1, interactive=False)

        def handle_download(selected_file):
            if selected_file:
                file_path = f"evaluation_dataset/output/{selected_file}"
                return file_path, f"檔案 {selected_file} 已準備下載。"
            return None, "請選擇有效的檔案進行下載。"

        download_button.click(
            handle_download,
            inputs=[output_file_selector],
            outputs=[gr.File(label="下載檔案"), download_status]
        )

        refresh_button.click(
            refresh_output_files,
            inputs=[],
            outputs=[output_file_selector, download_status]
        )

port = int(os.environ.get("PORT", 8080))
demo.launch(server_name="0.0.0.0", server_port=port)
