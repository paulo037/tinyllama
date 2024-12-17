from tinyllama import TinyLlama, load_model_weights, name_to_config
import argparse
from transformers import AutoTokenizer
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_text",   type=str,   required=True,  help="Input text to generate from." )
    parser.add_argument("--checkpoint",   type=str,   required=True,  help="Path to the model checkpoint.")
    parser.add_argument("--max_length",   type=int,   required=False, default=128, help="Maximum length of the generated sequence.")
    parser.add_argument("--sample",       type=bool,  required=False, default=True, help="If set to True, will sample the next token.")
    parser.add_argument("--model",        type=str,   required=False, default="tiny_LLaMA_1b",  help="Name of the model to be used.")
    parser.add_argument("--tokenizer_id", type=str,   required=False, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="HF_ID of the tokenizer to be used.")
    
    args = parser.parse_args()
    training_config = name_to_config[args.model]
    
    model = TinyLlama(training_config)
    
    if not load_model_weights(model, args.checkpoint):
        print("Failed to load model weights.")
        exit(1)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    
    input_ids = tokenizer(args.input_text, return_tensors="pt").input_ids.to(device)
    output = model.generate(input_ids, args.max_length, args.sample)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    
    


                          
