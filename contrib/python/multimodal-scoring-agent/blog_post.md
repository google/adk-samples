# Automating Visual Quality Assurance: Building a Multimodal Scoring Agent with Google ADK

Enterprises today are inundated with visual data, yet fully automating subjective visual tasks remains one of the hardest challenges in AI. Whether it’s grading a manufacturing defect on an assembly line, evaluating the layout of a submitted document, or assessing vehicle damage for an insurance claim, traditional computer vision solutions often fall short. They typically require massive, meticulously labeled datasets and expensive, custom-trained models for every specific use case. 

But what if you could just tell an AI your grading rubric and show it an image?

With the release of Gemini 2.5 Pro and the **Google Agent Development Kit (ADK)**, you can do exactly that. We recently contributed a new template to the [ADK samples repository](https://github.com/google/adk-samples) called the **Multimodal Scoring Agent**. This template demonstrates how to leverage Gemini's advanced multimodal reasoning alongside structured outputs to apply strict, JSON-defined scoring rubrics to images—reliably and deterministically.

## The Challenge of Subjective Visual Quality Assurance

Consider a quality assurance inspector at a circuit board manufacturing plant. They don't just look for "anomalies"; they follow a strict, codified set of rules (a rubric) to determine if a board passes, needs rework, or is a critical failure. 

Replicating this with standard generative AI often leads to inconsistent results—chatty responses, fluctuating grading scales, and unpredictable formatting. To integrate an AI inspector into an enterprise pipeline, we need **deterministic outputs** (a consistent data structure) and **traceable reasoning** (an explanation of *why* a grade was given based on the rubric).

## Enter the Multimodal Scoring Agent

Our newly merged ADK sample solves this by combining three powerful features:
1. **Gemini's Multimodal Understanding:** The ability to natively process and analyze high-resolution images.
2. **JSON-Defined Rubrics:** A dynamic way to inject domain-specific rules into the prompt.
3. **Structured Outputs:** Forcing the agent to reply strictly in a predefined schema (using Pydantic).

Let's walk through the code to see how easy it is to build this with the Google ADK.

### Step 1: Defining the Output Schema

First, we define exactly what we want the agent to return using Pydantic. By passing this schema to the ADK, we guarantee that the output will always contain a `grade`, `reasoning`, and `confidence` score—no more parsing messy text responses!

```python
from pydantic import BaseModel, Field


class ScoringResult(BaseModel):
    grade: str = Field(
        description="The deterministic grade assigned based on the rubric."
    )
    reasoning: str = Field(
        description="Detailed reasoning for why the grade was given, referencing specific criteria from the rubric."
    )
    confidence: float = Field(
        description="Confidence score in the assessment, from 0.0 to 1.0."
    )
```

### Step 2: Defining the Rubric

Instead of hardcoding rules into the prompt, we keep them in a clean JSON file (`rubric.json`). This separates the logic from the agent definition, allowing non-technical domain experts to easily update the grading criteria (e.g., changing from a "Pass/Fail" system to a Mensa-style scoring scale).

```json
{
  "rubric_name": "Circuit Board Quality Assurance",
  "criteria": [
    {
      "grade": "Pass",
      "conditions": ["All soldering joints are clean and shiny.", "No scorch marks."]
    },
    {
      "grade": "Fail - Critical",
      "conditions": ["Visible scorch marks or burning.", "Missing components."]
    }
  ],
  "instructions": "Evaluate the highest severity defect found."
}
```

### Step 3: Initializing the ADK Agent

Now, we wire it all together using the `Agent` class from Google ADK. Notice how we pass `ScoringResult` directly to the `response_schema` parameter. 

```python
from google.adk.agents import Agent
import json

# Initialize the Agent
scoring_agent = Agent(
    name="multimodal_scoring_agent",
    model="gemini-2.5-pro",  # Using Pro for advanced visual reasoning
    description="Evaluates images against a strict scoring rubric.",
    instruction="""You are an expert visual quality assurance inspector. 
You will be provided with an image and a JSON-defined scoring rubric. 
Analyze the image against the criteria. Apply the rules strictly and objectively. 
Return your final evaluation using the required structured output format.""",
    output_schema=ScoringResult,
)


def evaluate_image(image_path: str, rubric_path: str):
    import asyncio
    import uuid
    from google.genai import types
    from google.adk.runners import InMemoryRunner

    with open(rubric_path, "r") as f:
        rubric_data = json.load(f)

    prompt = f"Evaluate the attached image using this rubric:\n{json.dumps(rubric_data)}"

    with open(image_path, "rb") as f:
        img_data = f.read()

    content = types.Content(
        role="user",
        parts=[
            types.Part.from_bytes(data=img_data, mime_type="image/jpeg"),
            types.Part.from_text(text=prompt),
        ],
    )

    async def run_agent():
        runner = InMemoryRunner(agent=scoring_agent)
        runner.auto_create_session = True

        events = runner.run_async(
            user_id="local_user",
            session_id=str(uuid.uuid4()),
            new_message=content,
        )

        result = None
        async for event in events:
            if (
                hasattr(event, "message")
                and event.message
                and event.message.parts
            ):
                text = event.message.parts[0].text
                try:
                    result = ScoringResult.model_validate_json(text)
                except Exception:
                    pass
        return result

    return asyncio.run(run_agent())
```

### The Result

When you pass an image of a defective circuit board to this agent, it doesn't give you a conversational summary. It gives you an actionable, enterprise-ready data object:

```
=== Scoring Result ===
Grade: Fail - Critical
Confidence: 0.95
Reasoning:
Based on the Circuit Board Quality Assurance rubric, the image contains visible scorch marks near the main processor chip. This meets the criteria for a "Fail - Critical" grade as outlined in the conditions.
```

## Get Started Today

Automating subjective visual tasks no longer requires a six-month machine learning project. By combining the multimodal reasoning of Gemini with the structured, deterministic capabilities of the Google ADK, you can build reliable visual inspection agents in an afternoon.

Check out the full code in the [google/adk-samples repository](https://github.com/google/adk-samples) and try swapping out our manufacturing rubric with your own document layout rules, medical image assessments, or retail auditing guidelines!
