"""Configuration for Machine Learning Engineering Agent."""

import dataclasses
import os


@dataclasses.dataclass
class DefaultConfig:
    """Default configuration."""

    data_dir: str = "./machine_learning_engineering/tasks/"  # the directory path where the machine learning tasks and their data are stored.
    task_name: str = os.environ.get("TASK_NAME", "california-housing-prices")  # The name of the specific task to be loaded and processed.
    task_type: str = (
        "Tabular Regression"  # The type of machine learning problem.
    )
    lower: bool = True  # True if a lower value of the metric is better.
    workspace_dir: str = "./machine_learning_engineering/workspace/"  # Directory used for saving intermediate outputs, results, logs.
    agent_model: str = os.environ.get(
        "ROOT_AGENT_MODEL", "gemini-2.0-flash-001"
    )  # Name the LLM model to be used by the agent.
    task_description: str = ""  # The detailed description of the task.
    task_summary: str = ""  # The concise summary of the task.
    start_time: float = 0.0  # Timestamp indicating the start time of the task. Typically represented in seconds since the epoch.
    seed: int = 42  # The random seed value used to ensure reproducibility of experiments.
    exec_timeout: int = (
        600  # The maximum time in seconds allowed to complete the task.
    )
    num_solutions: int = 2  # The number of different solutions to generate or attempt for the given task.
    num_model_candidates: int = 2  # The number of different model architectures or hyperparameter sets to consider as candidates.
    max_retry: int = (
        10  # The maximum number of times to retry a failed operation.
    )
    max_debug_round: int = 5  # The maximum number of iterations or rounds allowed for the debugging step.
    max_rollback_round: int = 2  # The maximum number of times the system can rollback to a previous state, in case of errors or poor performance.
    inner_loop_round: int = 1  # The number of iterations or rounds to be executed within an inner loop of the system.
    outer_loop_round: int = 1  # The number of iterations or rounds to be executed within the outer loop, which might encompass multiple inner loops.
    ensemble_loop_round: int = 1  # The number of rounds or iterations dedicated to ensembling, combining multiple models or solutions.
    num_top_plans: int = 2  # The number of highest-scoring plans or strategies to select or retain.
    use_data_leakage_checker: bool = False  # Enable (`True`) or disable (`False`) a check for data leakage in the machine learning pipeline.
    use_data_usage_checker: bool = False  # Enable (`True`) or disable (`False`) a check for how data is being used, potentially for compliance or best practices.


CONFIG = DefaultConfig()
