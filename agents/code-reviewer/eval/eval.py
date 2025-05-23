from google.adk.evaluation.agent_evaluator import AgentEvaluator

import pathlib
import dotenv
import pytest

@pytest.fixture(scope="session", autouse=True)
def load_env():
    dotenv.load_dotenv()

def test_agent():
    AgentEvaluator.evaluate(
        agent_module="code_reviewer",
        eval_dataset_file_path_or_dir=str(
            pathlib.Path(__file__).parent / "agents/code-reviewer/eval/data/evalset002.evalset.json"
        ),
        num_runs=1
    )