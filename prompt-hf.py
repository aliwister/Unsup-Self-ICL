from argparse import ArgumentParser
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from statistics import mean
import torch, time, json
from tqdm import tqdm
import pandas as pd
from colorama import Fore, Style
import pdb
import os
import glob
import yaml

from experiment import Config
from promptparser import PromptParser
from task import TaskGenerator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {
    "llama-3-8B": ("meta-llama/Meta-Llama-3-8B-Instruct", 8192),
    "zephy7B": ("HuggingFaceH4/zephyr-7b-beta", 32768, 'auto'),
    "gemma-7B": ("google/gemma-7b", 8192, 'auto'),
    "Qwen-2-7B": ("Qwen/Qwen2-7B-Instruct", 32768),
    "mistral-7B": ("mistralai/Mistral-7B-Instruct-v0.3", 2048, 'auto'),
    "mixtral-7B": ("mistralai/Mixtral-8x7B-Instruct-v0.1", 2048, 'auto'),
    "openchat-8B": ("openchat/openchat-3.6-8b-20240522", 8192),
    "WizardLM-2-7B": ("lucyknada/microsoft_WizardLM-2-7B", 32768, 'auto'),
    "gpt-j-6b": ("EleutherAI/gpt-j-6b", 2048),
    "mol-t5-l": ('laituan245/molt5-large-smiles2caption', 512, 't5', False),
}



def write_pretty_json(file_path, data):
    import json
    with open(file_path, "w") as write_file:
        json.dump(data, write_file, indent=4)

def prepare_prompts(prompts, tokenizer, batch_size=4):
    batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
    batches_tok=[]
    tokenizer.padding_side="left"     
    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(
                prompt_batch, 
                return_tensors="pt", 
                padding='longest', 
                truncation=False, 
                pad_to_multiple_of=8,
                add_special_tokens=False).to("cuda") 
            )
    tokenizer.padding_side="right"
    return batches_tok

def main(config, model_name, input_dir, log_path, limit = -1, is_zero=True):
    self_icl_subdirs = ["demo-inputs", "demo-labels", "full-outputs"]
    cot_check = Fore.GREEN + "✔" + Style.RESET_ALL
    cot_cross = Fore.RED + "✘" + Style.RESET_ALL


    task_generator = TaskGenerator(
        task_input_path=config.task_input_path,
        task_desc_path=config.task_desc_path,
        batch_size=config.batch_size,
        verbose=True
    )
    for task_name in task_generator.task2desc.keys():
        task_log_path = log_path / task_name
        task_log_path.mkdir(parents=True, exist_ok=True)
        print("==================================================================================================================================")
        task = task_generator.get_task(task_name)

        if (is_zero):
            prompts_all = [f"Task description: {task.task_desc[0]}\n\nQ: {obj['input']}\nA:" for obj in task._samples]
        else:
            directory_path = f"{input_dir}/{task_name}/full-outputs"
            # Get all .txt files from the directory
            txt_files = glob.glob(os.path.join(directory_path, "*.txt"))

            # Initialize an empty list to store the contents of each file
            prompts = []
            idx = []
            # Loop through the list of files
            for file in txt_files:
                with open(file, 'r') as f:
                    file_content = f.read()  # Read the content of each .txt file
                    prompts.append("A:".join(file_content.split("A:")[:-1]) + "\nA:")
                    idx.append(int(os.path.splitext(os.path.basename(file))[0]))
                # Create a DataFrame where each row corresponds to the content of a file
            df = pd.DataFrame({'idx':idx, 'prompt': prompts, 'file': [os.path.basename(file) for file in txt_files]})
            df = df.sort_values(by="idx").reset_index(drop=True)
            prompts_all = df['prompt'].tolist()

        #pdb.set_trace()
        results = prompt(model_name, prompts_all)
        #pdb.set_trace()
        preds = [result.split("\n")[0].split(",")[0].strip("()").upper() for result in results]
        #print(preds)
        labels = [obj['target'] for obj in task._samples]
        npredict = task.sample_size
        ncorrect = sum(1 for result, reference in zip(preds, labels) if result.lower() == reference.lower())
        label_set_lower = {label.lower().strip("()") for label in task.label_set}
        nskip = sum(1 for pred in preds if pred.lower() not in label_set_lower)
        print(f"Correct count: {Fore.BLUE}{ncorrect}/{npredict} {Style.RESET_ALL}. Skipped={nskip}")

def prompt(model_name, prompts_all):
    accelerator = Accelerator()

    # load a base model and tokenizer
    model_path=models[model_name][0]
    model = AutoModelForCausalLM.from_pretrained(
        model_path,    
        device_map={"": accelerator.process_index},
        torch_dtype=torch.bfloat16,
        #cache_dir=f"../data/{model_name}"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path) #, cache_dir=f"../data/{model_name}")   
    tokenizer.pad_token = tokenizer.eos_token


    # sync GPUs and start the timer
    accelerator.wait_for_everyone()    
    start=time.time()

    # divide the prompt list onto the available GPUs 
    with accelerator.split_between_processes(prompts_all) as prompts:
        results=dict(outputs=[], num_tokens=0)

        # have each GPU do inference in batches
        prompt_batches=prepare_prompts(prompts, tokenizer, batch_size=2)
        pbar=tqdm(total=len(prompt_batches))    
        for prompts_tokenized in prompt_batches:
            outputs_tokenized=model.generate(
                **prompts_tokenized, 
                do_sample=True,
                temperature=.1,       # Control the randomness of the output
                top_p=0.9,             # Use nucleus sampling
                top_k=40,              # Use top-k sampling
                num_return_sequences=1,
                max_new_tokens=20,
                pad_token_id=tokenizer.eos_token_id)
            # remove prompt from gen. tokens
            outputs_tokenized=[ tok_out[len(tok_in):] 
                for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ] 

            # count and decode gen. tokens 
            num_tokens=sum([ len(t) for t in outputs_tokenized ])
            #pdb.set_trace()
            outputs=tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)
            processed_outputs = [output.split('#')[0].replace('"', '').strip() for output in outputs]


            # store in results{} to be gathered by accelerate
            results["outputs"].extend(processed_outputs)
            results["num_tokens"] += num_tokens
            time.sleep(0.1)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                pbar.update( accelerator.num_processes )

        results=[ results ] # transform to list, otherwise gather_object() will not collect correctly

    results_gathered=gather_object(results)

    if accelerator.is_main_process:
        print(len(results_gathered))
        timediff=time.time()-start
        num_tokens=sum([r["num_tokens"] for r in results_gathered ])
        #results_all = [item.split("#")[0] for r in results_gathered for item in r["outputs"]]
        results_all = [item for r in results_gathered for item in r["outputs"]]

        print(f"tokens/sec: {num_tokens//timediff}, time elapsed: {timediff}, num_tokens {num_tokens}")
        return results_all


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='/home/ali.lawati/Self-ICL/log/hf/stream/up-self-icl-llama')
    parser.add_argument('--model_name', type=str, default="mixtral-7B") 
    #parser.add_argument('--model_name', type=str, default="llama-3-8B") 
    parser.add_argument('--input_dir', type=str, default='/home/ali.lawati/Self-ICL/log/turbo/stream/jason') #, required=True)
    parser.add_argument('--config_path', type=Path, default=Path('./configs/self-icl.hf.yml')) #, required=True)

    parser.add_argument('--limit', type=int, default=100) 
    args = parser.parse_args() 

    config = Config(**yaml.safe_load(args.config_path.read_text()))

    log_path = Path(config.log_path) / config.inference_mode / config.exp_name
    log_path.mkdir(parents=True, exist_ok=True)
    main(config, args.model_name, args.input_dir, log_path, args.limit)