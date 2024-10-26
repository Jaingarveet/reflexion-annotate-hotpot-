from langchain.agents.react.base import DocstoreExplorer
from langchain.llms.base import BaseLLM

def reactLLMMock(prompt: str) -> str:
    """
    A mock function for a reaction-based LLM response. It simulates reasoning 
    and action generation based on the last line of the prompt.

    Args:
        prompt (str): The input prompt to the mock LLM.

    Returns:
        str: A simulated response based on the action inferred from the prompt.
    
    Raises:
        Exception: If the prompt does not contain a valid action type.
    """
    last_line = prompt.split('\n')[-1].strip()
    last_action = last_line.split(' ')[0].lower()

    if last_action == 'thought':
        # Simulates a reasoning step where the LLM decides to look up the "eastern sector".
        return 'It does not mention the eastern sector. So I need to look up eastern sector.'
    elif last_action == 'action':
        # Simulates an action decision where the LLM decides to perform a "Lookup" action.
        return 'Lookup[eastern sector]'
    else:
        raise Exception('Invalid action type')

def reflectLLMMock(prompt: str) -> str:
    """
    A mock function for a reflection-based LLM response. It simulates a reflection
    response based on the given prompt.

    Args:
        prompt (str): The input prompt to the mock reflection LLM.

    Returns:
        str: A simulated reflection response.
    """
    return "Last time I should have answered correctly"

class LLMMock(BaseLLM):
    """
    A mock class to simulate a language model (LLM) that can respond based on
    different types of prompts (e.g., reaction or reflection prompts).
    """
    def __init__(self):
        # Placeholder for initialization
        ...
    
    def __call__(self, prompt: str) -> str:
        """
        Determines which mock function to call based on the prompt type.

        Args:
            prompt (str): The input prompt.

        Returns:
            str: A simulated response based on the type of prompt.

        Raises:
            Exception: If the prompt does not contain a valid prefix.
        """
        if prompt.split('\n')[0].split(' ')[0] == 'Solve':
            return reactLLMMock(prompt)
        elif prompt.split('\n')[0].split(' ')[0] == 'You':
            return reflectLLMMock(prompt)
        else:
            raise Exception("Invalid LLM prompt")
    
    def get_num_tokens(self, text: str) -> int:
        """
        Returns the number of tokens in the given text. This mock implementation 
        always returns 0.

        Args:
            text (str): The input text to count tokens.

        Returns:
            int: Always returns 0, as this is a mock function.
        """
        return 0
    
class DocStoreExplorerMock(DocstoreExplorer):
    """
    A mock class to simulate the DocstoreExplorer functionality, allowing the agent
    to perform search and lookup actions in a simplified manner.
    
    Attributes:
        summary (str): A mock summary for the search results.
        body (str): A mock detailed response for the lookup results.
    """
    
    def __init__(self):
        """
        Initializes the DocStoreExplorerMock with predefined search and lookup results.
        """
        self.summary = "The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas."
        self.body = "(Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny."
    
    def search(self, search: str, sents: int = 5) -> str:
        """
        Simulates a search action in the docstore, returning a mock summary.

        Args:
            search (str): The search query.
            sents (int): The number of sentences to retrieve (not used in the mock).

        Returns:
            str: A predefined summary response.
        """
        return self.summary
    
    def lookup(self, term: str) -> str:
        """
        Simulates a lookup action in the docstore, returning a mock detailed response.

        Args:
            term (str): The term to look up.

        Returns:
            str: A predefined detailed response.
        """
        return self.body