import re
import string
from typing import Tuple

import gym
from langchain import Wikipedia
from langchain.agents.react.base import DocstoreExplorer

class QAEnv(gym.Env):
    """
    A custom reinforcement learning environment for a question-answering task using Gym.
    The agent interacts with a DocstoreExplorer (e.g., Wikipedia) to search and look up information,
    aiming to find the correct answer within a limited number of steps.

    Attributes:
        question (str): The question the agent needs to answer.
        key (str): The correct answer used to verify agent responses.
        max_steps (int): The maximum steps allowed for the agent to find the answer.
        explorer (DocstoreExplorer): An explorer object for searching and looking up information.
        curr_step (int): The current step count in the episode.
        terminated (bool): Indicates if the episode has ended.
        answer (str): The agent's answer to the question.
    """
    
    def __init__(self,
                 question: str,
                 key: str,
                 max_steps: int = 6,
                 explorer: DocstoreExplorer = DocstoreExplorer(Wikipedia())):
        """
        Initializes the QA environment with a question, answer key, maximum steps, and explorer.

        Args:
            question (str): The question for the agent to answer.
            key (str): The correct answer to validate the agent's answer.
            max_steps (int): The maximum number of steps allowed to find the answer.
            explorer (DocstoreExplorer): Tool to interact with Wikipedia for searches and lookups.
        """
        self.question = question
        self.key = key
        self.max_steps = max_steps
        self.explorer = explorer

        self.reset()

    def reset(self):
        """
        Resets the environment for a new episode. Sets step counter to zero, 
        marks the episode as not terminated, and clears the answer.
        """
        self.curr_step = 0
        self.terminated = False
        self.answer = ''

    def step(self, action: str) -> Tuple[str, bool, bool, bool, bool]:
        """
        Takes an action (e.g., search, lookup, finish) and returns the observation, 
        reward, and episode status.

        Args:
            action (str): Action taken by the agent, e.g., "Search[topic]" or "Finish[answer]".

        Returns:
            Tuple: Observation (str), reward (bool), terminated (bool), truncated (bool), current step (int).
        """
        action_type, argument = parse_action(action)

        if action_type == 'Finish':
            self.answer = argument
            if self.is_correct():
                observation = 'Answer is CORRECT'
            else: 
                observation = 'Answer is INCORRECT'
            self.terminated = True

        elif action_type == 'Search':
            try:
                observation = self.explorer.search(argument).strip('\n').strip()
            except Exception as e:
                print(e)
                observation = f'Could not find that page, please try again.'
                    
        elif action_type == 'Lookup':
            try:
                observation = self.explorer.lookup(argument).strip('\n').strip()
            except ValueError:
                observation = f'The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.'

        else:
            observation = 'Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].'

        reward = self.is_correct()
        terminated = self.is_terminated()
        truncated = self.is_truncated()

        self.curr_step += 1

        return observation, reward, terminated, truncated, self.curr_step

    def is_correct(self) -> bool:
        """
        Checks if the agent's answer matches the correct key.

        Returns:
            bool: True if the agent's answer matches the key, else False.
        """
        return EM(self.answer, self.key)
    
    def is_terminated(self) -> bool:
        """
        Checks if the episode has been manually terminated.

        Returns:
            bool: True if terminated, else False.
        """
        return self.terminated

    def is_truncated(self) -> bool:
        """
        Checks if the agent has reached the maximum number of steps.

        Returns:
            bool: True if the step limit has been reached, else False.
        """
        return self.curr_step >= self.max_steps

def parse_action(string: str) -> Tuple[str, str]:
    """
    Parses an action string to identify the action type and argument.

    Args:
        string (str): The action string, e.g., "Search[topic]" or "Finish[answer]".

    Returns:
        Tuple[str, str]: A tuple of (action_type, argument) if successful, else (None, None).
    """
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)
    
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument
    else:
        return None, None

def normalize_answer(s: str) -> str:
    """
    Normalizes an answer by removing articles, punctuation, whitespace, and converting to lowercase.

    Args:
        s (str): The answer string to normalize.

    Returns:
        str: The normalized answer.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def EM(answer: str, key: str) -> bool:
    """
    Checks if the answer exactly matches the key after normalization.

    Args:
        answer (str): The answer given by the agent.
        key (str): The correct answer.

    Returns:
        bool: True if the normalized answer matches the key, False otherwise.
    """
    return normalize_answer(answer) == normalize_answer(key)