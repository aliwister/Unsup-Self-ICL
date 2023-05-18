import math
import json
import yaml
import pandas as pd
from typing import List
from pathlib import Path
from argparse import ArgumentParser, Namespace
from colorama import Fore, Style
from dataclasses import dataclass
from task import TaskGenerator
from model import Model
from prompt import StreamPrompt, BatchPrompt, Shot
from promptparser import PromptParser

@dataclass
class Config(object):
    exp_name: str
    # paths
    bbh_input_path: str
    bbh_task_desc_path: str
    log_path: str
    # experiment settings
    inference_mode: str # "stream" or "batch"
    exemplars_mode: str # "self-icl" or "standard"
    num_demos: int # self-icl: number of pseudo-demos to generate; standard: number of real demos to use (0: zero-shot, >0: few-shot a.k.a. standrad ICL)
    use_cot: bool # whether to use chain-of-thought
    label_method: str # "self" (LLM-generated) or "random" (randomly sample from the label space) -> only used when exemplars_mode == "self-icl"
    diverse_exemplars: bool # whether to generate diverse exemplars -> only used when exemplars_mode == "self-icl"
    # sizes
    batch_size: int # only used when inference_mode == "batch"
    test_sample_size: int
    # model hparams
    model: str # e.g., text-davinci-003
    max_tokens: int
    temperature: float
    demo_temperature: float # temperature when generating pseudo-demos (only used when exemplars_mode == "self-icl")
    top_p: float

class Experiment(object):
    self_icl_subdirs = ["demo-inputs", "demo-labels", "full-outputs"]
    
    def __init__(self, config: Config) -> None:
        print(f"Initializing experiment {config.exp_name}...")
        self._config = config
        self._model = Model(
            Namespace(
                model=self._config.model,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                top_p=self._config.top_p,
            )
        )
        # ensure correct batch_size
        if self._config.inference_mode == "stream":
            self._config.batch_size = 1
        elif self._config.inference_mode == "batch":
            assert self._config.batch_size > 1
        else:
            raise ValueError(f"Invalid inference_mode: {self._config.inference_mode}")
        # ensure correct exemplars_mode
        if self._config.exemplars_mode not in ["self-icl", "standard"]:
            raise ValueError(f"Invalid exemplars_mode: {self._config.exemplars_mode}")
        # make logging path
        self._log_path = Path(self._config.log_path) / self._config.inference_mode / self._config.exp_name
        self._log_path.mkdir(parents=True, exist_ok=True)
        (self._log_path / "config.yml").write_text(yaml.dump(vars(self._config)))
        # make prompt parser
        self._prompt_parser = PromptParser(num_demos=self._config.num_demos)
        
    def print_configs(self) -> None:
        print("Configs:")
        for k, v in vars(self._config).items():
            print(f"\t{k}: {v}")
            
    def run(
        self,
        task_continue_from: str = None,
        sample_start_from: int = 0,
        label_type: str = None
    ) -> None:
        if label_type and (label_type not in set(TaskGenerator.task2label_type.values())):
            raise ValueError(f"Invalid label_type: {label_type}")
        
        self.print_configs()

        task_generator = TaskGenerator(
            task_input_path=self._config.bbh_input_path,
            task_desc_path=self._config.bbh_task_desc_path,
            batch_size=self._config.batch_size,
            verbose=True
        )
        continue_flag = True if task_continue_from else False
        failed_cases = [] # list of (task_name, sample_idx)
        for task_name in TaskGenerator.task2label_type.keys():
            if continue_flag and (task_name != task_continue_from):
                continue
            continue_flag = False
            
            # make task log dir
            task_log_path = self._log_path / task_name
            task_log_path.mkdir(parents=True, exist_ok=True)
            if self._config.exemplars_mode == "self-icl":
                # make pseudo-demos log dir
                for subdir in self.self_icl_subdirs:
                    (task_log_path / subdir).mkdir(parents=True, exist_ok=True)
            # start running task
            if label_type and (TaskGenerator.task2label_type[task_name] != label_type):
                print(f"Skipping task {task_name} with label_type {task.label_type}...")
                continue
            
            task = task_generator.get_task(task_name)
            if task.label_type == "class":
                label_set = task.label_set
            else:
                label_set = None
            
            task.set_counter(sample_start_from)
            num_runs = math.ceil(self._config.test_sample_size / self._config.batch_size)
            for i in range(num_runs):
                if i < sample_start_from:
                    print(f"Skipping sample #{i}...")
                    continue
                print(f"Running sample #{i}:")
                task_inputs = task.get_new_inputs()
                shots = []
                # TODO: get bbh-hand-made shots standard few-shot setting
                if (self._config.exemplars_mode == "standard") and (self._config.num_demos > 0):
                    raise NotImplementedError
                # prepare initial prompt
                if self._config.inference_mode == "stream":
                    prompt = StreamPrompt(
                        task_desc=task.task_desc,
                        inputs=task_inputs,
                        num_demos=self._config.num_demos,
                        shots=shots
                    )
                else: # batch
                    prompt = BatchPrompt(
                        task_desc=task.task_desc,
                        inputs=task_inputs,
                        num_demos=self._config.num_demos,
                        shots=shots
                    )
                # augment prompt with pseudo-demos if "self-icl"
                if self._config.exemplars_mode == "self-icl":
                    # generate pseudo-demos
                    # inputs
                    demo_prompt = prompt.gen_demo_inputs(diversity=self._config.diverse_exemplars)
                    demo_prompt, demo_inputs = self._model.complete(demo_prompt, label_set=None, temperature=self._config.demo_temperature)
                    full_demo_inputs = demo_prompt + demo_inputs
                    (task_log_path / "demo-inputs" / f"{i}.txt").write_text(full_demo_inputs)
                    # parse demo inputs to separate instances
                    try:
                        sep_demo_inputs = self._prompt_parser.split_demo_inputs(full_demo_inputs)
                    except ValueError:
                        print(Fore.RED + f"Task {task_name} sample #{i} failed: failed to parse demo inputs" + Style.RESET_ALL)
                        failed_cases.append([task_name, i])
                        continue
                    # labels
                    shots = []
                    if self._config.inference_mode == "stream":
                        for j, sep_demo_input in enumerate(sep_demo_inputs):
                            sep_demo_prompt = StreamPrompt(
                                task_desc=task.task_desc,
                                inputs=sep_demo_input,
                                num_demos=0, # NOTE
                                shots=[]
                            ).gen_prediction()
                            print(f"Predicting demo #{j} ->", end='')
                            sep_demo_prompt, sep_demo_label = self._model.complete(sep_demo_prompt, label_set, temperature=self._config.temperature)
                            if sep_demo_prompt[-1] == '(':
                                sep_demo_label = '(' + sep_demo_label
                            shot = Shot(_input=sep_demo_input, _label=sep_demo_label.strip())
                            shots.append(shot)
                            # logging
                            print(sep_demo_label)
                            (task_log_path / "demo-labels" / f"{i}-{j}.txt").write_text(str(shot))
                        # update prompt to augmented prompt
                        prompt = StreamPrompt(
                            task_desc=task.task_desc,
                            inputs=task_inputs,
                            num_demos=self._config.num_demos,
                            shots=shots
                        )
                    else: # batch
                        sep_demo_prompt = BatchPrompt(
                            task_desc=task.task_desc,
                            inputs=sep_demo_inputs,
                            num_demos=0, # NOTE
                            shots=shots
                        )        
                        raise NotImplementedError
                        # update prompt to augmented prompt
                    
                # run inference
                print(f"Predicting sample #{i} ->", end='')
                pred_prompt = prompt.gen_prediction()
                pred_prompt, res_text = self._model.complete(pred_prompt, label_set, temperature=self._config.temperature)
                print(res_text)
                # save results
                full_text = pred_prompt + res_text
                if self._config.exemplars_mode == "standard":
                    (task_log_path / f"{i}.txt").write_text(full_text)
                else: # self-icl
                    (task_log_path / "full-outputs" / f"{i}.txt").write_text(full_text)
        
        # save failed cases
        if len(failed_cases) > 0:
            print(f"Saving {len(failed_cases)} failed cases...")
            (self._log_path / "failed_cases.json").write_text(json.dumps(failed_cases, indent=4))
                    
    def evaluate(
        self,
        label_type: str = None
    ) -> None:
        # for generating task labels
        task_gen = TaskGenerator(
            task_input_path=self._config.bbh_input_path,
            task_desc_path=self._config.bbh_task_desc_path,
            batch_size=1, # evaluate one by one during evaluation
            verbose=True
        )
        # start evaluation
        total_correct = 0
        total_predict = 0
        eval_results = dict()
        for task_name in TaskGenerator.task2label_type.keys():
            if label_type and (TaskGenerator.task2label_type[task_name] != label_type):
                print(f"Skipping task {task_name} with label_type {task.label_type}...")
                continue
            task = task_gen.get_task(task_name)
            task_log_path = self._log_path / task_name
            
            ncorrect = 0
            for i in range(self._config.test_sample_size):
                # read inference result
                if self._config.exemplars_mode == "standard":
                    full_res = (task_log_path / f"{i}.txt").read_text()
                else: # self-icl
                    full_res = (task_log_path / "full-outputs" / f"{i}.txt").read_text()
                # parse inference result
                label = task.get_new_labels().strip("()").upper()
                pred = self._prompt_parser.extract_pred(full_res).upper()
                print(f"Sample #{i}: label = {label}, pred = {pred} -> ", end='')
                if label == pred:
                    print(Fore.GREEN + "✔")
                    ncorrect += 1
                else:
                    print(Fore.RED + "✘")
                print(Style.RESET_ALL, end='')
            
            eval_results[task_name] = {
                "ncorrect": ncorrect,
                "total": self._config.test_sample_size,
                "accuracy": ncorrect / self._config.test_sample_size
            }
            print(f"Correct count: {Fore.BLUE}{ncorrect}/{self._config.test_sample_size}{Style.RESET_ALL}")
            total_correct += ncorrect
            total_predict += self._config.test_sample_size
        
        print(f"{self._config.exp_name} -> Total correct count: {Fore.BLUE}{total_correct}/{total_predict}{Style.RESET_ALL}; Accuracy: {Fore.BLUE}{total_correct / total_predict * 100:.2f}%{Style.RESET_ALL}")
        # save evaluation results
        df = pd.DataFrame(eval_results)
        df.to_csv(self._log_path / f"eval_results_{self._config.test_sample_size}-testsize.csv", index_label="Item")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--task_continue_from", type=str, default=None)
    parser.add_argument("--sample_start_from", type=int, default=0)
    parser.add_argument("--label_type", type=str, default=None)
    parser.add_argument("--eval", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = Config(**yaml.safe_load(args.config_path.read_text()))
    experiment = Experiment(config)
    
    if args.eval:
        experiment.evaluate(label_type=args.label_type)
    else:
        experiment.run(
            task_continue_from=args.task_continue_from,
            sample_start_from=args.sample_start_from,
            label_type=args.label_type
        )
