import os
import openai
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap



def generate_answers_batch(questions, list_relevant_documents):
    """
    Hàm này xử lý batch các câu hỏi và các tài liệu liên quan.
    Args:
        questions (list): Danh sách các câu hỏi.
        list_relevant_documents (list): Danh sách các ngữ cảnh tương ứng với từng câu hỏi.
    
    Returns:
        list: Danh sách các câu trả lời cho mỗi câu hỏi.
    """
    # Define template
    template = """Trả lời câu hỏi bằng một câu đầy đủ, dựa trên ngữ cảnh được cung cấp sau đây:
    {context}
    Câu hỏi: {question}
    """
    
    # Prompt template
    prompt = ChatPromptTemplate.from_template(template)

    # Output parser
    output_parser = StrOutputParser()

    # Function để gọi OpenAI GPT-4 API
    def call_gpt4(question, context):
        prompt_text = template.format(context=context, question=question)
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Sử dụng GPT-4 API
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=200,  # Giới hạn số từ trong câu trả lời
            temperature=0.1  # Mức độ sáng tạo thấp
        )
        answer = response['choices'][0]['message']['content']
        return answer

    # Tạo batch data cho các câu hỏi và tài liệu liên quan
    batch_data = [{"question": question, "relevant_document": relevant}
                  for question, relevant in zip(questions, list_relevant_documents)]

    # Xử lý batch data với GPT-4 API
    answers = []
    for item in batch_data:
        answer = call_gpt4(item['question'], item['relevant_document'])
        answers.append(answer)
    
    return answers

