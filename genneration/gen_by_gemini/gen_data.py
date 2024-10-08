import json
from create_prompt import create_new_format, create_prompt_format
from function import generate_answers_batch
# from time import time
import time
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":

    # # Path to json data
    path_data_qa = "/point/namnt/DATN/retrieval/data/data_test.json"
    path_template = "/point/namnt/DATN/retrieval/data/template.json"

    # Convert to format for gen prompt
    data_qa = json.load(open(path_data_qa, "r", encoding="utf-8"))
    data_template = json.load(open(path_template, "r", encoding="utf-8"))

    data_new_template = create_new_format(data_template)
    list_question, list_context = create_prompt_format(data_qa, data_new_template)

    # Gen with batch_size
    batch_size = 50
    output_file = '/point/namnt/DATN/genneration/data/data_test.csv'

    # Khởi tạo file CSV với header nếu nó chưa tồn tại
    # Sử dụng mode='w' để ghi tiêu đề chỉ lần đầu tiên
    df = pd.DataFrame(columns=["question", "context", "answer"])
    df.to_csv(output_file, mode='w', index=False)

    for i in tqdm(range(0, len(list_question), batch_size)):
        batch_question = list_question[i:i + batch_size]
        batch_context = list_context[i:i + batch_size]

        try:
            output = generate_answers_batch(batch_question, batch_context)
            print(f"output: {output}")

            # Chuẩn bị batch hiện tại thành DataFrame
            batch_output = []
            for j in range(len(batch_question)):
                tmp = {
                    "question": batch_question[j],
                    "context": batch_context[j],
                    "answer": output[j]
                }
                batch_output.append(tmp)

            # Convert batch_output to DataFrame và lưu vào file CSV (append mode)
            df_batch = pd.DataFrame(batch_output)
            df_batch.to_csv(output_file, mode='a', header=False, index=False)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)

        time.sleep(60)
