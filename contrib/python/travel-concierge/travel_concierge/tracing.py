import os
import warnings

from arize.otel import register
from dotenv import load_dotenv
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from opentelemetry import trace

load_dotenv()


def instrument_adk_with_arize() -> trace.Tracer:
    """Instrument the ADK with Arize."""

    if os.getenv("ARIZE_SPACE_ID") is None:
        warnings.warn("ARIZE_SPACE_ID is not set", stacklevel=2)
        return None
    if os.getenv("ARIZE_API_KEY") is None:
        warnings.warn("ARIZE_API_KEY is not set", stacklevel=2)
        return None

    tracer_provider = register(
        space_id=os.getenv("ARIZE_SPACE_ID"),
        api_key=os.getenv("ARIZE_API_KEY"),
        project_name=os.getenv("ARIZE_PROJECT_NAME", "adk-travel-concierge"),
    )

    GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)

    return tracer_provider.get_tracer(__name__)
