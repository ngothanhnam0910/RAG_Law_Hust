import json
import os


def create_new_format(template):
    
    """_summary_
    Chuyển về dạng dict      
    Ex:  {"học bổng_0": "title + text"}  
    """
    
    final_dict = {}
    
    for field in template:
        field_id = field["Field_id"]
        for info in field["infor"]:
            update_name = (field_id + "_" + info["infor_id"]).strip()
            update_text = info["title"] + "." + info["text"]
            final_dict[update_name] = update_text
            
    # with open(path_new_template , 'w', encoding='utf-8') as f:
    #     json.dump(final_dict, f, ensure_ascii=False, indent=4)
        
    return final_dict

def create_prompt_format(qa, data_new_template):
    list_question = []
    list_context = []

    for pair in qa['items']:
        update_field = (pair["relevant_info"][0]["Field_id"] + "_" + pair["relevant_info"][0]["infor_id"]).strip()
        list_question.append(pair["question"])
        list_context.append(data_new_template[update_field])

    return list_question, list_context

            
                


