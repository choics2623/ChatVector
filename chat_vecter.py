import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def main(chat_path, pt_path, cpt_path, save_path):
    tokenizer = AutoTokenizer.from_pretrained(chat_path)
    model = AutoModelForCausalLM.from_pretrained(pt_path, torch_dtype=torch.bfloat16, device_map='cpu')
    model_chat = AutoModelForCausalLM.from_pretrained(chat_path, torch_dtype=torch.bfloat16, device_map='cpu')
    cpt_model = AutoModelForCausalLM.from_pretrained(cpt_path, torch_dtype=torch.bfloat16, device_map='cpu')
    
    model_state = model.state_dict()
    model_chat_state = model_chat.state_dict()
    cpt_model_state = cpt_model.state_dict()
    
    weight_diff = {}
    
    for key in model_state.keys():
        weight_diff[key] = model_chat_state[key] - model_state[key]
        cpt_model_state[key] = cpt_model_state[key] + weight_diff[key]
    
    cpt_model.load_state_dict(cpt_model_state)
    cpt_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chat_path', type=str, required=True, help='Chat 모델 경로')
    parser.add_argument('--pt_path', type=str, required=True, help='PT 모델 경로')
    parser.add_argument('--cpt_path', type=str, required=True, help='CPT 모델 경로')
    parser.add_argument('--save_path', type=str, required=True, help='저장할 경로')
    
    args = parser.parse_args()
    
    main(args.chat_path, args.pt_path, args.cpt_path, args.save_path)
