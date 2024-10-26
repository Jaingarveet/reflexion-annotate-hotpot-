import re, string, os
from typing import List, Union, Literal
from enum import Enum
import tiktoken
from langchain import OpenAI, Wikipedia
from langchain.llms.base import BaseLLM
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langchain.agents.react.base import DocstoreExplorer
from langchain.docstore.base import Docstore
from langchain.prompts import PromptTemplate
from llm import AnyOpenAILLM
from prompts import reflect_prompt, react_agent_prompt, react_reflect_agent_prompt, REFLECTION_HEADER, LAST_TRIAL_HEADER, REFLECTION_AFTER_LAST_TRIAL_HEADER
from prompts import cot_agent_prompt, cot_reflect_agent_prompt, cot_reflect_prompt, COT_INSTRUCTION, COT_REFLECT_INSTRUCTION
from fewshots import WEBTHINK_SIMPLE6, REFLECTIONS, COT, COT_REFLECT

class ReflexionStrategy(Enum):
    """
    Enum class for reflection strategies.

    Defines the types of reflection strategies available for the agent:
    
    - NONE: No reflection.
    - LAST_ATTEMPT: Use the last reasoning trace in the context.
    - REFLEXION: Apply reflexion to the next reasoning trace.
    - LAST_ATTEMPT_AND_REFLEXION: Use the last reasoning trace in the context and apply reflexion to the next trace.
    """
    NONE = 'base'
    LAST_ATTEMPT = 'last_trial' 
    REFLEXION = 'reflexion'
    LAST_ATTEMPT_AND_REFLEXION = 'last_trial_and_reflexion'


class CoTAgent:
    """
    Chain-of-Thought (CoT) Agent for reasoning tasks with self-reflection.

    Attributes:
        question (str): The question for the agent to answer.
        context (str): Additional context for answering the question.
        key (str): Expected answer for evaluation.
        agent_prompt (PromptTemplate): Prompt template for agent reasoning.
        reflect_prompt (PromptTemplate): Prompt template for agent reflection.
        cot_examples (str): Examples to guide the agent's reasoning.
        reflect_examples (str): Examples for reflection.
        self_reflect_llm (AnyOpenAILLM): LLM used for generating reflections.
        action_llm (AnyOpenAILLM): LLM used for generating actions in the reasoning.
        reflections (List[str]): List of reflections generated during reasoning.
        reflections_str (str): Formatted string of reflections.
        answer (str): Agent's current answer.
        step_n (int): Current step count.
    """
    
    def __init__(self,
                    question: str,
                    context: str,
                    key: str,
                    agent_prompt: PromptTemplate = cot_reflect_agent_prompt,
                    reflect_prompt: PromptTemplate = cot_reflect_prompt,
                    cot_examples: str = COT,
                    reflect_examples: str = COT_REFLECT,
                    self_reflect_llm: AnyOpenAILLM = AnyOpenAILLM(
                                            temperature=0,
                                            max_tokens=250,
                                            model_name="gpt-3.5-turbo",
                                            model_kwargs={"stop": "\n"},
                                            openai_api_key=os.environ['OPENAI_API_KEY']),
                    action_llm: AnyOpenAILLM = AnyOpenAILLM(
                                            temperature=0,
                                            max_tokens=250,
                                            model_name="gpt-3.5-turbo",
                                            model_kwargs={"stop": "\n"},
                                            openai_api_key=os.environ['OPENAI_API_KEY']),
                    ) -> None:
        self.question = question
        self.context = context
        self.key = key
        self.agent_prompt = agent_prompt
        self.reflect_prompt = reflect_prompt
        self.cot_examples = cot_examples 
        self.reflect_examples = reflect_examples
        self.self_reflect_llm = self_reflect_llm
        self.action_llm = action_llm
        self.reflections: List[str] = []
        self.reflections_str = ''
        self.answer = ''
        self.step_n: int = 0
        self.reset()

    def run(self,
            reflexion_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION) -> None:
        """
        Runs the reasoning process using a specified reflexion strategy.

        This method resets the scratchpad, then generates an answer by stepping through
        reasoning stages. If the answer is incorrect and a reflexion strategy is provided,
        it applies reflexion before re-running.

        Args:
            reflexion_strategy (ReflexionStrategy): The reflexion strategy to use.
        """
        if self.step_n > 0 and not self.is_correct() and reflexion_strategy != ReflexionStrategy.NONE:
            self.reflect(reflexion_strategy)
        self.reset()
        self.step()
        self.step_n += 1

    def step(self) -> None:
        """
        Executes one reasoning step (think-act-observe).

        This method generates a thought, performs an action, and then observes the outcome. 
        It checks if the action leads to a correct answer or if further steps are required.
        """
        # Think
        self.scratchpad += f'\nThought:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        action_type, argument = parse_action(action)
        print(self.scratchpad.split('\n')[-1])  

        self.scratchpad += f'\nObservation: '
        if action_type == 'Finish':
            self.answer = argument
            if self.is_correct():
                self.scratchpad += 'Answer is CORRECT'
            else: 
                self.scratchpad += 'Answer is INCORRECT'
            self.finished = True
            return
        else:
            print('Invalid action type, please try again.')
    
    def reflect(self,
                strategy: ReflexionStrategy) -> None:
        """
        Runs the reflection process based on the specified strategy.

        Based on the strategy, the method applies a certain type of reflection to 
        improve future responses.

        Args:
            strategy (ReflexionStrategy): Reflection strategy to apply.
        """
        print('Running Reflexion strategy...')
        if strategy == ReflexionStrategy.LAST_ATTEMPT:
            self.reflections = [self.scratchpad]
            self.reflections_str = format_last_attempt(self.question , self.reflections[0])
        elif strategy == ReflexionStrategy.REFLEXION:
            self.reflections += [self.prompt_reflection()]
            self.reflections_str = format_reflections(self.reflections)
        elif strategy == ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION:
            self.reflections_str = format_last_attempt(self.question , self.scratchpad)
            self.reflections = [self.prompt_reflection()]
            self.reflections_str += '\n'+ format_reflections(self.reflections, header = REFLECTION_AFTER_LAST_TRIAL_HEADER)
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
        print(self.reflections_str)
    
    def prompt_reflection(self) -> str:
        """
        Generates a reflection prompt to encourage better future responses.

        Returns:
            str: Reflection response generated by the LLM.
        """
        return format_step(self.self_reflect_llm(self._build_reflection_prompt()))

    def reset(self) -> None:
        """
        Resets the agent's state for a new reasoning attempt.
        """
        self.scratchpad: str = ''
        self.finished = False

    def prompt_agent(self) -> str:
        """
        Prompts the agent LLM for reasoning or action steps.

        Returns:
            str: Agent's response generated by the LLM.
        """
        return format_step(self.action_llm(self._build_agent_prompt()))
    
    def _build_agent_prompt(self) -> str:
        """
        Constructs the prompt for agent LLM based on the question, context, reflections, and scratchpad.

        Returns:
            str: Formatted agent prompt string.
        """
        return self.agent_prompt.format(
                            examples = self.cot_examples,
                            reflections = self.reflections_str,
                            context = self.context,
                            question = self.question,
                            scratchpad = self.scratchpad)

    def _build_reflection_prompt(self) -> str:
        """
        Constructs the reflection prompt based on the scratchpad and examples.

        Returns:
            str: Formatted reflection prompt string.
        """
        return self.reflect_prompt.format(
                            examples = self.reflect_examples,
                            context = self.context,
                            question = self.question,
                            scratchpad = self.scratchpad)
 
    def is_finished(self) -> bool:
        """
        Checks if the agent has finished reasoning.

        Returns:
            bool: True if reasoning is complete, False otherwise.
        """
        return self.finished

    def is_correct(self) -> bool:
        """
        Checks if the agent's answer is correct.

        Returns:
            bool: True if answer matches the expected key, False otherwise.
        """
        return EM(self.answer, self.key)   

class ReactAgent:
    """
    Reactive Agent that performs a sequence of thought-action-observation steps 
    to answer a question. It leverages an LLM to determine the best action 
    at each step, and can access external knowledge sources like Wikipedia.

    Attributes:
        question (str): The question the agent is trying to answer.
        answer (str): The final answer generated by the agent.
        key (str): The expected correct answer for validation.
        max_steps (int): Maximum number of steps before halting.
        agent_prompt (PromptTemplate): Prompt template for the agent's actions.
        react_examples (str): Example prompts to guide the agent's reasoning.
        docstore (DocstoreExplorer): Interface for searching and looking up information.
        llm (AnyOpenAILLM): LLM instance used for generating actions.
        enc (tiktoken.Encoding): Encoder for managing token limits.
        scratchpad (str): Log of the agent's thoughts, actions, and observations.
        step_n (int): Current step number in the reasoning sequence.
        finished (bool): Indicates whether the agent has completed its task.
    """

    def __init__(self,
                 question: str,
                 key: str,
                 max_steps: int = 6,
                 agent_prompt: PromptTemplate = react_agent_prompt,
                 docstore: Docstore = Wikipedia(),
                 react_llm: AnyOpenAILLM = AnyOpenAILLM(
                                            temperature=0,
                                            max_tokens=100,
                                            model_name="gpt-3.5-turbo",
                                            model_kwargs={"stop": "\n"},
                                            openai_api_key=os.environ['OPENAI_API_KEY']),
                 ) -> None:
        """
        Initializes the ReactAgent with necessary parameters and resets its state.

        Args:
            question (str): The question for the agent to answer.
            key (str): Expected answer for evaluation.
            max_steps (int): Maximum allowed steps for the reasoning sequence.
            agent_prompt (PromptTemplate): Template for agent prompts.
            docstore (Docstore): Knowledge source for search and lookup actions.
            react_llm (AnyOpenAILLM): LLM used for generating responses.
        """
        self.question = question
        self.answer = ''
        self.key = key
        self.max_steps = max_steps
        self.agent_prompt = agent_prompt
        self.react_examples = WEBTHINK_SIMPLE6

        self.docstore = DocstoreExplorer(docstore)  # Interface for knowledge base
        self.llm = react_llm
        self.enc = tiktoken.encoding_for_model("text-davinci-003")
        self.__reset_agent()

    def run(self, reset=True) -> None:
        """
        Runs the agent's reasoning process until it either halts or finishes.

        Args:
            reset (bool): If True, resets the agent's state before starting.
        """
        if reset:
            self.__reset_agent()
        
        while not self.is_halted() and not self.is_finished():
            self.step()

    def step(self) -> None:
        """
        Executes a single thought-action-observation cycle.

        In each cycle:
        - Think: Generate a thought about the question.
        - Act: Take an action based on the thought.
        - Observe: Record observations based on the action outcome.
        """
        # Think
        self.scratchpad += f'\nThought {self.step_n}:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        action_type, argument = parse_action(action)
        print(self.scratchpad.split('\n')[-1])

        # Observe
        self.scratchpad += f'\nObservation {self.step_n}: '
        
        if action_type == 'Finish':
            self.answer = argument
            if self.is_correct():
                self.scratchpad += 'Answer is CORRECT'
            else: 
                self.scratchpad += 'Answer is INCORRECT'
            self.finished = True
            self.step_n += 1
            return

        if action_type == 'Search':
            try:
                self.scratchpad += format_step(self.docstore.search(argument))
            except Exception as e:
                print(e)
                self.scratchpad += 'Could not find that page, please try again.'
            
        elif action_type == 'Lookup':
            try:
                self.scratchpad += format_step(self.docstore.lookup(argument))
            except ValueError:
                self.scratchpad += 'The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.'

        else:
            self.scratchpad += 'Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].'

        print(self.scratchpad.split('\n')[-1])

        self.step_n += 1

    def prompt_agent(self) -> str:
        """
        Generates a response from the LLM based on the current state.

        Returns:
            str: Response generated by the LLM.
        """
        return format_step(self.llm(self._build_agent_prompt()))
    
    def _build_agent_prompt(self) -> str:
        """
        Constructs the prompt for the agent based on the current scratchpad and examples.

        Returns:
            str: Formatted agent prompt.
        """
        return self.agent_prompt.format(
                            examples=self.react_examples,
                            question=self.question,
                            scratchpad=self.scratchpad)
    
    def is_finished(self) -> bool:
        """
        Checks if the agent has finished reasoning.

        Returns:
            bool: True if reasoning is complete, False otherwise.
        """
        return self.finished

    def is_correct(self) -> bool:
        """
        Checks if the agent's answer is correct by comparing it to the key.

        Returns:
            bool: True if the answer matches the key, False otherwise.
        """
        return EM(self.answer, self.key)

    def is_halted(self) -> bool:
        """
        Determines if the agent should halt due to exceeding max steps or token limits.

        Returns:
            bool: True if the agent should halt, False otherwise.
        """
        return ((self.step_n > self.max_steps) or (len(self.enc.encode(self._build_agent_prompt())) > 3896)) and not self.finished

    def __reset_agent(self) -> None:
        """
        Resets the agent's state, clearing the scratchpad and initializing counters.
        """
        self.step_n = 1
        self.finished = False
        self.scratchpad: str = ''

    def set_qa(self, question: str, key: str) -> None:
        """
        Sets a new question and expected answer key for the agent.

        Args:
            question (str): New question for the agent.
            key (str): Expected answer key for validation.
        """
        self.question = question
        self.key = key

class ReactReflectAgent(ReactAgent):
    """
    Extended Reactive Agent with Reflection capability, allowing it to self-reflect on past reasoning 
    sequences to improve its responses in future steps. This agent is designed to handle both reactive 
    actions and reflective reasoning to enhance its accuracy over time.

    Attributes:
        reflect_llm (AnyOpenAILLM): LLM instance for generating reflective responses.
        reflect_prompt (PromptTemplate): Template for reflection prompts.
        reflect_examples (str): Example reflections to guide the agent's reasoning.
        reflections (List[str]): List of reflections generated during the reasoning process.
        reflections_str (str): Formatted string of all reflections.
    """

    def __init__(self,
                 question: str,
                 key: str,
                 max_steps: int = 6,
                 agent_prompt: PromptTemplate = react_reflect_agent_prompt,
                 reflect_prompt: PromptTemplate = reflect_prompt,
                 docstore: Docstore = Wikipedia(),
                 react_llm: AnyOpenAILLM = AnyOpenAILLM(
                                             temperature=0,
                                             max_tokens=100,
                                             model_name="gpt-3.5-turbo",
                                             model_kwargs={"stop": "\n"},
                                             openai_api_key=os.environ['OPENAI_API_KEY']),
                 reflect_llm: AnyOpenAILLM = AnyOpenAILLM(
                                               temperature=0,
                                               max_tokens=250,
                                               model_name="gpt-3.5-turbo",
                                               openai_api_key=os.environ['OPENAI_API_KEY']),
                 ) -> None:
        """
        Initializes the Reflective Agent with question, answer key, reflection LLM, and prompt templates.

        Args:
            question (str): The question for the agent to answer.
            key (str): Expected answer for validation.
            max_steps (int): Maximum allowed steps for the reasoning sequence.
            agent_prompt (PromptTemplate): Template for agent prompts.
            reflect_prompt (PromptTemplate): Template for reflection prompts.
            docstore (Docstore): Knowledge source for search and lookup actions.
            react_llm (AnyOpenAILLM): LLM used for generating reactive responses.
            reflect_llm (AnyOpenAILLM): LLM used for generating reflective responses.
        """
        super().__init__(question, key, max_steps, agent_prompt, docstore, react_llm)
        self.reflect_llm = reflect_llm
        self.reflect_prompt = reflect_prompt
        self.reflect_examples = REFLECTIONS
        self.reflections: List[str] = []
        self.reflections_str: str = ''
    
    def run(self, reset=True, reflect_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION) -> None:
        """
        Runs the agent's reasoning process with optional reflection based on the strategy.

        Args:
            reset (bool): If True, resets the agent's state before starting.
            reflect_strategy (ReflexionStrategy): Strategy defining how the agent should reflect on its reasoning.
        """
        if (self.is_finished() or self.is_halted()) and not self.is_correct():
            self.reflect(reflect_strategy)

        ReactAgent.run(self, reset)
    
    def reflect(self, strategy: ReflexionStrategy) -> None:
        """
        Performs reflection based on the provided strategy. This helps the agent improve future responses
        by analyzing past actions and adjusting reasoning patterns.

        Args:
            strategy (ReflexionStrategy): Reflection strategy to be used.
        """
        print('Reflecting...')
        if strategy == ReflexionStrategy.LAST_ATTEMPT:
            self.reflections = [self.scratchpad]
            self.reflections_str = format_last_attempt(self.question, self.reflections[0])
        elif strategy == ReflexionStrategy.REFLEXION: 
            self.reflections += [self.prompt_reflection()]
            self.reflections_str = format_reflections(self.reflections)
        elif strategy == ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION: 
            self.reflections_str = format_last_attempt(self.question, self.scratchpad)
            self.reflections = [self.prompt_reflection()]
            self.reflections_str += format_reflections(self.reflections, header=REFLECTION_AFTER_LAST_TRIAL_HEADER)
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
        print(self.reflections_str)
    
    def prompt_reflection(self) -> str:
        """
        Prompts the reflection LLM to generate a reflective response based on the current state.

        Returns:
            str: Reflective response generated by the reflection LLM.
        """
        return format_step(self.reflect_llm(self._build_reflection_prompt()))

    def _build_reflection_prompt(self) -> str:
        """
        Constructs the reflection prompt based on past examples and current scratchpad content.

        Returns:
            str: Formatted reflection prompt for the LLM.
        """
        return self.reflect_prompt.format(
                            examples=self.reflect_examples,
                            question=self.question,
                            scratchpad=truncate_scratchpad(self.scratchpad, tokenizer=self.enc))
 
    def _build_agent_prompt(self) -> str:
        """
        Constructs the agent prompt incorporating reflections and scratchpad content.

        Returns:
            str: Formatted agent prompt for the LLM.
        """
        return self.agent_prompt.format(
                            examples=self.react_examples,
                            reflections=self.reflections_str,
                            question=self.question,
                            scratchpad=self.scratchpad)


### String Utilities ###
gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")

def parse_action(string):
    """
    Parses the action from the agent's output to determine action type and arguments.

    Args:
        string (str): Output string from the agent to parse.

    Returns:
        tuple: (action_type, argument) if match is found, otherwise None.
    """
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)
    
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument
    else:
        return None

def format_step(step: str) -> str:
    """
    Formats a single step by removing unnecessary newlines and extra spaces.

    Args:
        step (str): The step text to format.

    Returns:
        str: Formatted step text.
    """
    return step.strip('\n').strip().replace('\n', '')

def format_reflections(reflections: List[str], header: str = REFLECTION_HEADER) -> str:
    """
    Formats reflections by adding a header and listing each reflection.

    Args:
        reflections (List[str]): List of reflection strings.
        header (str): Header for the reflection section.

    Returns:
        str: Formatted reflections text.
    """
    if reflections == []:
        return ''
    else:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

def format_last_attempt(question: str, scratchpad: str, header: str = LAST_TRIAL_HEADER):
    """
    Formats the last attempt by adding a header, question, and truncated scratchpad.

    Args:
        question (str): The question that was asked.
        scratchpad (str): Log of thoughts, actions, and observations.
        header (str): Header for the last attempt section.

    Returns:
        str: Formatted last attempt text.
    """
    return header + f'Question: {question}\n' + truncate_scratchpad(scratchpad, tokenizer=gpt2_enc).strip('\n').strip() + '\n(END PREVIOUS TRIAL)\n'

def truncate_scratchpad(scratchpad: str, n_tokens: int = 1600, tokenizer=gpt2_enc) -> str:
    """
    Truncates the scratchpad text if it exceeds the specified token limit by removing the largest observations.

    Args:
        scratchpad (str): Text of the scratchpad to truncate.
        n_tokens (int): Maximum allowed tokens for the scratchpad.
        tokenizer (Tokenizer): Tokenizer for calculating token counts.

    Returns:
        str: Truncated scratchpad text.
    """
    lines = scratchpad.split('\n')
    observations = filter(lambda x: x.startswith('Observation'), lines)
    observations_by_tokens = sorted(observations, key=lambda x: len(tokenizer.encode(x)))
    while len(gpt2_enc.encode('\n'.join(lines))) > n_tokens:
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        lines[ind] = largest_observation.split(':')[0] + ': [truncated wikipedia excerpt]'
    return '\n'.join(lines)

def normalize_answer(s):
    """
    Normalizes an answer by removing articles, punctuation, and extra whitespace, and converting to lowercase.

    Args:
        s (str): The answer string to normalize.

    Returns:
        str: Normalized answer string.
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

def EM(answer, key) -> bool:
    """
    Checks if an answer matches the expected key using exact match after normalization.

    Args:
        answer (str): The answer generated by the agent.
        key (str): The expected correct answer.

    Returns:
        bool: True if the normalized answer matches the key, False otherwise.
    """
    return normalize_answer(answer) == normalize_answer(key)
