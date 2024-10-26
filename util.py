import os
import joblib

def summarize_trial(agents):
    """
    Summarizes the trial results by categorizing agents as correct or incorrect.

    Args:
        agents (list): A list of agent objects that have completed the trial.

    Returns:
        Tuple: Lists of agents that produced correct answers and those that finished but were incorrect.
    """
    correct = [a for a in agents if a.is_correct()]  # Agents with correct answers
    incorrect = [a for a in agents if a.is_finished() and not a.is_correct()]  # Finished agents with incorrect answers
    return correct, incorrect

def remove_fewshot(prompt: str) -> str:
    """
    Removes the few-shot examples section from the prompt for cleaner logging.

    Args:
        prompt (str): The complete prompt including examples.

    Returns:
        str: The prompt without the few-shot examples section.
    """
    prefix = prompt.split('Here are some examples:')[0]
    suffix = prompt.split('(END OF EXAMPLES)')[1]
    return prefix.strip('\n').strip() + '\n' + suffix.strip('\n').strip()

def log_trial(agents, trial_n):
    """
    Logs the trial results, organizing agents by correct and incorrect answers.

    Args:
        agents (list): A list of agent objects that have completed the trial.
        trial_n (int): The trial number for labeling.

    Returns:
        str: A formatted log string for the trial, detailing correct and incorrect agents.
    """
    correct, incorrect = summarize_trial(agents)  # Summarize trial results

    log = f"""
########################################
BEGIN TRIAL {trial_n}
Trial summary: Correct: {len(correct)}, Incorrect: {len(incorrect)}
#######################################
"""

    # Log correct agents' prompts and answers
    log += '------------- BEGIN CORRECT AGENTS -------------\n\n'
    for agent in correct:
        log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'

    # Log incorrect agents' prompts and answers
    log += '------------- BEGIN INCORRECT AGENTS -----------\n\n'
    for agent in incorrect:
        log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'

    return log

def summarize_react_trial(agents):
    """
    Summarizes the trial results for a React trial by categorizing agents as correct, halted, or incorrect.

    Args:
        agents (list): A list of agent objects that have completed the React trial.

    Returns:
        Tuple: Lists of agents that produced correct answers, halted, or finished incorrectly.
    """
    correct = [a for a in agents if a.is_correct()]  # Agents with correct answers
    halted = [a for a in agents if a.is_halted()]  # Agents that halted during the trial
    incorrect = [a for a in agents if a.is_finished() and not a.is_correct()]  # Finished agents with incorrect answers
    return correct, incorrect, halted

def log_react_trial(agents, trial_n):
    """
    Logs the results for a React trial, organizing agents by correct, incorrect, and halted statuses.

    Args:
        agents (list): A list of agent objects that have completed the React trial.
        trial_n (int): The trial number for labeling.

    Returns:
        str: A formatted log string for the React trial, detailing correct, incorrect, and halted agents.
    """
    correct, incorrect, halted = summarize_react_trial(agents)  # Summarize trial results

    log = f"""
########################################
BEGIN TRIAL {trial_n}
Trial summary: Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}
#######################################
"""

    # Log correct agents' prompts and answers
    log += '------------- BEGIN CORRECT AGENTS -------------\n\n'
    for agent in correct:
        log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'

    # Log incorrect agents' prompts and answers
    log += '------------- BEGIN INCORRECT AGENTS -----------\n\n'
    for agent in incorrect:
        log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'

    # Log halted agents' prompts and answers
    log += '------------- BEGIN HALTED AGENTS -----------\n\n'
    for agent in halted:
        log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'

    return log

def save_agents(agents, dir: str):
    """
    Saves each agent's state to a file in the specified directory.

    Args:
        agents (list): A list of agent objects to save.
        dir (str): The directory path where the agents will be saved.
    """
    os.makedirs(dir, exist_ok=True)  # Create directory if it doesn't exist
    for i, agent in enumerate(agents):
        joblib.dump(agent, os.path.join(dir, f'{i}.joblib'))  # Save each agent as a .joblib file