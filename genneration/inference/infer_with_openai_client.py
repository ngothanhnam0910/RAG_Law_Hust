from openai import OpenAI
from datasets import Dataset, load_dataset
import time

eval_dataset = load_dataset("json", data_files="test_dataset.json", split="train")
# print(eval_dataset[7]["messages"][:2])
# exit()


# init the client but point it to TGI
client = OpenAI(
    base_url="http://localhost:8080/v1/",
    api_key="-"
)

messages_list = []
for i in range(50):
    messages = eval_dataset[i]["messages"][:2]
    messages_list.append(messages)
    
t1 = time.time()
for messages in messages_list:
    chat_completion = client.chat.completions.create(
        model="/workspace",
        messages=messages,  # Gửi từng danh sách
        stream=True,
        max_tokens=1024
    )

    # for chunk in chat_completion:
    #     print(chunk.choices[0].delta.content)
        
t2 = time.time()
print(f"Total time: {t2 - t1}")
# t1 = time.time()
# chat_completion = client.chat.completions.create(
#     model="/workspace",
#     messages=eval_dataset[7]["messages"][:2],
#     stream=True,
#     max_tokens = 1024,
#     frequency_penalty=0.5,  # Thay đổi giá trị này theo ý muốn
#     temperature = 0,
# )

# t2 = time.time()

# Khởi tạo một danh sách để lưu trữ các chunk
# response_chunks = []

# for chunk in chat_completion:
#     # Thêm nội dung vào danh sách
#     response_chunks.append(chunk.choices[0].delta.content)

# # Nối tất cả các chunk lại thành một câu
# full_response = ''.join(response_chunks)
# print(full_response.split("system")[0].strip())