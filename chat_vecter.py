import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main(chat_path, pt_path, cpt_path, save_path, is_lora, upload_path):
    tokenizer = AutoTokenizer.from_pretrained(chat_path)
    model = AutoModelForCausalLM.from_pretrained(pt_path, torch_dtype=torch.bfloat16, device_map='cpu')
    model_chat = AutoModelForCausalLM.from_pretrained(chat_path, torch_dtype=torch.bfloat16, device_map='cpu')
    if is_lora:
        model = PeftModel.from_pretrained(model, cpt_path)
        cpt_model = model.merge_and_unload()
    else:
        cpt_model = AutoModelForCausalLM.from_pretrained(cpt_path, torch_dtype=torch.bfloat16, device_map='cpu')
    
    model_state = model.state_dict()
    model_chat_state = model_chat.state_dict()
    cpt_model_state = cpt_model.state_dict()
    
    weight_diff = {}
    
    for key in model_state.keys():
        weight_diff[key] = model_chat_state[key] - model_state[key]
        cpt_model_state[key] = cpt_model_state[key] + weight_diff[key]
    
    cpt_model.load_state_dict(cpt_model_state)
    
    if save_path:
        cpt_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    if upload_path:
        cpt_model.push_to_hub(upload_path, private=True)
        tokenizer.push_to_hub(upload_path, private=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chat_path', type=str, required=True, help='Chat 모델 경로')
    parser.add_argument('--pt_path', type=str, required=True, help='PT 모델 경로')
    parser.add_argument('--cpt_path', type=str, required=True, help='CPT 모델 경로')
    parser.add_argument('--save_path', type=str, required=False, help='저장할 경로')
    parser.add_argument('--lora', action='store_true', help='LoRA 모델 사용 여부')

    parser.add_argument('--upload_path', type=str, required=False, help='hf 업로드 경로')
    
    
    args = parser.parse_args()
    
    main(args.chat_path, args.pt_path, args.cpt_path, args.save_path, args.lora, args.upload_path)
