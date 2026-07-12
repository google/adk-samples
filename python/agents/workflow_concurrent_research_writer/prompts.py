"""Prompts for the ADK Workflow Demo."""

JOIN_AND_DISTILL_PROMPT = """
You are a research analyst. Your task is to synthesize a list of research summaries {DistillInput.research}, provided as input, into a single, coherent report.
The user's original topic was: {topic}.

The research summaries are provided as your input. Please begin synthesizing them now.
"""

GENERATE_BLOG_POST_PROMPT = """
As a content creator, write a blog post that expands on the provided thesis, which will be given as your input.
Use the research report below to substantiate your points. Include an attention-grabbing sentence that links to another platform where the article 
might be posted.

Research Report:
{research_report}
"""
