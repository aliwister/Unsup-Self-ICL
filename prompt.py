import abc
import unittest
from typing import List, Union
from dataclasses import dataclass

@dataclass
class Shot(object):
    _input: str
    _label: str
    
    @property
    def input(self) -> str:
        return self._input
    
    @property
    def label(self) -> str:
        return self._label
    
    def __repr__(self) -> str:
        return f"Q: {self._input}\nA: {self._label}"

# Currently only implement "stream" prompts
class Prompt(metaclass=abc.ABCMeta):
    cot_prompt = " Let's think step by step."
    
    def __init__(
        self,
        task_desc: str,
        inputs: Union[str, List[str]],
        num_demos: int,
        shots: List[Shot] = [],
        shots_raw: List[str] = [],
        is_up = False
    ) -> None:
        self._task_desc = task_desc
        self._inputs = inputs
        self._num_demos = num_demos
        self._shots = shots
        self._shots_raw = shots_raw
        self._is_up = is_up
    
    @abc.abstractmethod  
    def gen_prediction(self, cot: bool = False) -> str:
        return NotImplemented

    @abc.abstractmethod
    def gen_demo_inputs(self, diversity: bool = False) -> str:
        return NotImplemented
    
class StreamPrompt(Prompt):

    def __init__(
        self,
        task_desc: str,
        inputs: str,
        num_demos: int,
        shots: List[Shot] = [],
        shots_raw: List[str] = [],
        is_up = False
    ) -> None:
        super().__init__(task_desc, inputs, num_demos, shots, shots_raw, is_up)

    def gen_prediction(self, cot: bool = False, add_parenthesis: bool = False) -> str:
        """
        ### Prompting format:
        Task description: [task description].

        Q: [pseudo-demo-input 1]
        A: [pseudo-demo-label 1]

        ...

        Q: [pseudo-demo-input n]
        A: [pseudo-demo-label n]

        Q:
        """
        # task description
        prompt = [f"Task description: {self._task_desc[0]}\n\n"]
        if cot:
            prompt.append("Format:\n")
            prompt.append('Starting with "Therefore, the correct answer is ..." before giving your final answer.\n')
            prompt.append("If options are availbale, you must pick one as the final answer.\n")
            prompt.append("It's very important that you stick to the format.\n\n")
        # in-context examples
        if (self._is_up):
            prompt.append(f"{self._task_desc[1]}:\n")
            for shot in self._shots_raw:
                prompt.append(f"Q: {shot}\n")
            prompt.append('\n###\n')
            prompt.append(f"{self._task_desc[2]}:\n\n")
        for shot in self._shots:
            prompt.append(f"Q: {shot.input}\n")
            prompt.append(f"A:{self.cot_prompt if cot else ''} {shot.label}\n\n")
        # current input
        prompt.append(f"Q: {self._inputs}\n")
        prompt.append(f"A:{self.cot_prompt if cot else ''}")
        return "".join(prompt)
    
    def gen_demo_inputs(self, diversity: Union[bool, str] = False) -> str:
        """
        ### Prompting format:
        Following is an example instance for the task: [task description]. Please come up with [num_shot] new[diverse_prompt] instances for the task.
        Example instance:
        [test input]

        New instance 1:
        """
        if type(diversity) == bool:
            diverse_prompt = ", diverse, and creative" if diversity else ""
            return f"Following is an example instance for the task: {self._task_desc[0]} Please come up with {self._num_demos} new{diverse_prompt} instances for the task.\n\nExample instance:\nQ: {self._inputs.split('Options')[0]}\n\nNew instance 1:\nQ:"
        elif (type(diversity) == str) and (diversity == "no-new"):
            return f"Following is an example instance for the task: {self._task_desc[0]} Please come up with {self._num_demos} instances for the task.\nExample instance:\nQ: {self._inputs.split('Options')[0]}\n\nInstance 1:\nQ:"


class BatchPrompt(Prompt):
    
    def __init__(
        self,
        task_desc: str,
        inputs: List[str],
        num_demos: int,
        shots: Union[str, List[Shot]] = None
    ) -> None:
        super().__init__(task_desc, inputs, num_demos, shots)
        
    def gen_prediction(self, cot: bool = False, add_parenthesis: bool = False) -> str:
        """
        ### Prompting format:
        Task description: [task description]. Please answer the following questions one-by-one.

        Q1: [pseudo-demo-input 1]
        ...
        Q[NUM_SHOT]: [pseudo-demo-input NUM_SHOT]
        Q[NUM_SHOT + 1]: [test input 1]
        ...
        Q[NUM_SHOT + BATCH_SIZE]: [test input BATCH_SIZE]

        A1: [pseudo-demo-label 1]
        ...
        A[NUM_SHOT]: [pseudo-demo-label NUM_SHOT]
        A[NUM_SHOT + 1]:
        """
        prompt = [f"Task description: {self._task_desc} Please answer the following questions one-by-one.\n\n"]
        # TODO: add CoT prompts
        # add in-context examples
        if self._shots:
            if type(self._shots) == str: # batched shots
                answer_start = self._shots.index("\nA1:")
                Qs = self._shots[:answer_start]
                As = self._shots[answer_start:]
                prompt.append(Qs)        
            else:
                raise NotImplementedError
        # current input questions
        for i, input_ in enumerate(self._inputs, start=1):
            prompt.append(f"Q{self._num_demos + i}: {input_}\n")
        if self._shots:
            prompt.append(As)
        prompt.append(f"\nA{self._num_demos + 1}:" + (" (" if add_parenthesis else ""))
        return "".join(prompt)
    
    def gen_demo_inputs(self, diversity: bool = False) -> str:
        """
        Following are [BATCH_SIZE] exapmle instances for the task: [TASK_DESCRIPTION]. Please come up with [NUM_SHOT] new, diverse, and creative instances for the task.
        Example instance 1:
        Q: [TEST_INPUT_1]

        Example instance 2:
        Q: [TEST_INPUT_2]

        ...

        Example instance [BATCH_SIZE]:
        Q: [TEST_INPUT_[BATCH_SIZE]]

        New instance 1:
        Q:
        """
        diverse_prompt = ", diverse, and creative" if diversity else ""
        prompt = [f"Following are {len(self._inputs)} example instances for the task: {self._task_desc} Please come up with {self._num_demos} new{diverse_prompt} instances for the task.\n"]
        # example instances
        for i, input_ in enumerate(self._inputs):
            prompt.append(f"Example instance {i+1}:\n")
            prompt.append(f"Q: {input_}\n\n")
        # new instances
        prompt.append("New instance 1:\n")
        prompt.append("Q:")
        return "".join(prompt)


class TestStreamPrompt(unittest.TestCase):

    def setUp(self):
        self.task_desc = "Evaluate the result of a random Boolean expression."
        self.inputs = "not ( True ) and ( True ) is"
        self.num_demos = 3
        self.zero_shots = []
        self.few_shots = [
            Shot("True and not not ( not False ) is", "True"),
            Shot("not True or False or ( False ) is", "False"),
            Shot("False or not ( True ) and False is", "False")
        ]
        
        self.zs_prompt = StreamPrompt(self.task_desc, self.inputs, self.num_demos, self.zero_shots)
        self.fs_prompt = StreamPrompt(self.task_desc, self.inputs, self.num_demos, self.few_shots)
    
    def test_gen_prediction(self):
        self.assertEqual(
            self.zs_prompt.gen_prediction(),
            "Task description: Evaluate the result of a random Boolean expression.\n\nQ: not ( True ) and ( True ) is\nA:"
        )
        self.assertEqual(
            self.fs_prompt.gen_prediction(),
            "Task description: Evaluate the result of a random Boolean expression.\n\nQ: True and not not ( not False ) is\nA: True\n\nQ: not True or False or ( False ) is\nA: False\n\nQ: False or not ( True ) and False is\nA: False\n\nQ: not ( True ) and ( True ) is\nA:"
        )
    
    def test_gen_demo_inputs(self):
        self.assertEqual(
            self.zs_prompt.gen_demo_inputs(),
            "Following is an example instance for the task: Evaluate the result of a random Boolean expression. Please come up with 3 new instances for the task.\nExample instance:\nQ: not ( True ) and ( True ) is\n\nNew instance 1:\nQ:"
        )
        self.assertEqual(
            self.fs_prompt.gen_demo_inputs(),
            "Following is an example instance for the task: Evaluate the result of a random Boolean expression. Please come up with 3 new instances for the task.\nExample instance:\nQ: not ( True ) and ( True ) is\n\nNew instance 1:\nQ:"
        )


# for running unit tests
if __name__ == "__main__":
    # print out the prompt
    task_desc = "Clarify the meaning of sentences with ambiguous pronouns."
    inputs = "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\nSentence: The patient was referred to the specialist because he had a rare skin condition.\nOptions:\n(A) The patient had a skin condition\n(B) The specialist had a skin condition\n(C) Ambiguous"
    num_demos = 3
    
    prompt = StreamPrompt(task_desc, inputs, num_demos)
    print(prompt.gen_prediction())
    print(prompt.gen_demo_inputs(diversity=True))
    
    # unittest
    print("\nRunning unit tests...")
    unittest.main()
    
