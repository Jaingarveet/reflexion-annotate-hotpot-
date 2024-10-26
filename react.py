import os
from typing import List
import dotenv

import gym
import tiktoken
from langchain import OpenAI
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate

from environment import QAEnv
from prompts import reflect_prompt, react_agent_prompt, react_reflect_agent_prompt, REFLECTION_HEADER
from fewshots import WEBTHINK_SIMPLE6, REFLECTIONS

# Load environment variables
dotenv.load_dotenv()

class ReactAgent:
    """
    A question-answering ReAct Agent that uses a Large Language Model (LLM) to answer 
    questions through iterative thought and action steps. Interacts with an environment
    and evaluates answers based on the LLM's reasoning.

    Attributes:
        question (str): The input question to be answered.
        env (QAEnv): The environment where actions are taken and observations are made.
        agent_prompt (PromptTemplate): The prompt template guiding the agent's responses.
        react_llm (BaseLLM): The LLM responsible for generating thought and action responses.
    """
    def __init__(self,
                 question: str,
                 env: QAEnv,
                 agent_prompt: PromptTemplate = react_agent_prompt,
                 react_llm: BaseLLM = OpenAI(
                                             temperature=0,
                                             max_tokens=100,
                                             model_name="text-davinci-003",
                                             model_kwargs={"stop": "\n"},
                                             openai_api_key=os.environ['OPENAI_API_KEY']),
                 ) -> None:
        
        self.question = question
        self.agent_prompt = agent_prompt
        self.react_examples = WEBTHINK_SIMPLE6

        self.env = env
        self.env.reset()
        self.reset()
        self.truncated, self.reward, self.terminated = False, False, False

        self.llm = react_llm
        self.enc = tiktoken.encoding_for_model("text-davinci-003")

    def run(self, reset=True) -> None:
        """
        Runs the agent by continuously stepping through actions until the answer
        is either correct, truncated, or terminated.

        Args:
            reset (bool): Whether to reset the environment before running.
        """
        if reset:
            self.env.reset()
            self.reset()
        
        while not (self.is_truncated() or self.is_terminated()):
            self.step()
    
    def step(self) -> None:
        """
        Executes a single step for the agent:
        - Thinks by generating a thought.
        - Acts by deciding on an action.
        - Observes the result from the environment.
        """
        # Think
        self.scratchpad += f'\nThought {self.curr_step}:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction {self.curr_step}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        print(self.scratchpad.split('\n')[-1])

        # Observe
        self.scratchpad += f'\nObservation {self.curr_step}: '
        observation, self.reward, self.terminated, self.truncated, self.curr_step = self.env.step(action)
        self.scratchpad += observation
        print(self.scratchpad.split('\n')[-1])

    def prompt_agent(self) -> str:
        """
        Prompts the LLM to generate a response based on the current agent prompt.

        Returns:
            str: The generated response after formatting.
        """
        return format_step(self.llm(self._build_agent_prompt()))
    
    def _build_agent_prompt(self) -> str:
        """
        Builds the prompt for the agent by combining examples, the question, and 
        the scratchpad (thought process so far).

        Returns:
            str: The constructed prompt for the agent's thought process.
        """
        return self.agent_prompt.format(
                            examples=self.react_examples,
                            question=self.question,
                            scratchpad=self.scratchpad)
    
    def is_terminated(self) -> bool:
        """
        Checks if the agent has reached a termination condition in the environment.

        Returns:
            bool: True if terminated, otherwise False.
        """
        return self.env.is_terminated()

    def is_correct(self) -> bool:
        """
        Checks if the agent's answer is correct.

        Returns:
            bool: True if the answer is correct, otherwise False.
        """
        return self.env.is_correct()

    def is_truncated(self) -> bool:
        """
        Checks if the agent has been truncated based on step limit or token length.

        Returns:
            bool: True if truncated, otherwise False.
        """
        return self.env.is_truncated() or (len(self.enc.encode(self._build_agent_prompt())) > 3896)

    def reset(self) -> None:
        """
        Resets the agent's scratchpad and current step counter.
        """
        self.scratchpad = ''
        self.curr_step = 1


class ReactReflectAgent(ReactAgent):
    """
    A Self-Reflecting React Agent that reflects on its past actions and attempts to
    improve responses in subsequent runs.

    Attributes:
        reflect_llm (BaseLLM): The LLM responsible for generating reflection responses.
        reflect_prompt (PromptTemplate): The prompt template for guiding reflection.
        reflections (List[str]): A list of reflections generated by the agent.
    """
    def __init__(self,
                 question: str,
                 env: QAEnv,
                 agent_prompt: PromptTemplate = react_reflect_agent_prompt,
                 reflect_prompt: PromptTemplate = reflect_prompt,
                 react_llm: BaseLLM = OpenAI(
                                             temperature=0,
                                             max_tokens=100,
                                             model_name="text-davinci-003",
                                             model_kwargs={"stop": "\n"},
                                             openai_api_key=os.environ['OPENAI_API_KEY']),
                 reflect_llm: BaseLLM = OpenAI(
                                               temperature=0,
                                               max_tokens=250,
                                               model_name="text-davinci-003",
                                               openai_api_key=os.environ['OPENAI_API_KEY']),
                 ) -> None:
        
        super().__init__(question, env, agent_prompt, react_llm)
        self.reflect_llm = reflect_llm
        self.reflect_prompt = reflect_prompt
        self.reflect_examples = REFLECTIONS
        self.reflections = []
    
    def run(self, reset=True) -> None:
        """
        Runs the Reflect Agent with the ability to reflect before re-attempting if
        the previous answer was incorrect.

        Args:
            reset (bool): Whether to reset the environment before running.
        """
        if (self.is_terminated() or self.is_truncated()) and not self.is_correct():
            self.reflect()

        ReactAgent.run(self, reset)
    
    def reflect(self) -> None:
        """
        Appends a new reflection by calling the reflection prompt.
        """
        self.reflections.append(self.prompt_reflection())
    
    def prompt_reflection(self) -> str:
        """
        Generates a reflection using the reflection LLM.

        Returns:
            str: The generated reflection after formatting.
        """
        return format_step(self.reflect_llm(self._build_reflection_prompt()))

    def _build_reflection_prompt(self) -> str:
        """
        Builds the reflection prompt by combining examples, question, and the scratchpad.

        Returns:
            str: The constructed reflection prompt for the agent.
        """
        return self.reflect_prompt.format(
                            examples=self.reflect_examples,
                            question=self.question,
                            scratchpad=self._format_scratchpad())
    
    def _build_agent_prompt(self) -> str:
        """
        Builds the prompt for the agent by combining examples, reflections, question, 
        and the scratchpad.

        Returns:
            str: The constructed prompt for the agent's thought process.
        """
        return self.agent_prompt.format(
                            examples=self.react_examples,
                            reflections=format_reflections(self.reflections),
                            question=self.question,
                            scratchpad=self.scratchpad)
    
    def _format_scratchpad(self) -> str:
        """
        Truncates the scratchpad to fit within a token limit, preserving the most
        relevant lines of thought.

        Returns:
            str: The truncated and formatted scratchpad text.
        """
        lines = self.scratchpad.split('\n')
        lines_by_tokens = sorted(lines, key=lambda x: len(self.enc.encode(x)))
        while len(self.enc.encode('\n'.join(lines))) > 1600:
            ind = lines.index(lines_by_tokens.pop(-1))
            line = lines[ind]
            lines[ind] = line.split(':')[0] + ': ...'
        return '\n'.join(lines)
    
    

### String Operations ###
def format_reflections(reflections: List[str]) -> str:
    """
    Formats a list of reflections by joining them with a header.

    Args:
        reflections (List[str]): The reflections to format.

    Returns:
        str: The formatted reflections as a single string.
    """
    if reflections == []:
        return ''
    else:
        header = REFLECTION_HEADER
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

def format_step(step: str) -> str:
    """
    Formats a single step by trimming whitespace and removing newlines.

    Args:
        step (str): The step text to format.

    Returns:
        str: The formatted step text.
    """
    return step.strip('\n').strip().replace('\n', '')