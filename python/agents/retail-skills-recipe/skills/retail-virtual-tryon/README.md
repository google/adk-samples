# Retail Virtual Try-On

Virtual try-on agent using Gemini image generation (flash/pro tiers) for clothing, eyewear, jewelry, cosmetics, and footwear. Includes a pre-flight product-cutout classifier and configurable safety levels.

## Prerequisites

- Python 3.10+
- Google Cloud project with Vertex AI APIs enabled
- [retail-product-search](../retail-product-search/) set up first

## Setup

```bash
pip install -r requirements.txt
python scripts/setup_tryon.py --config assets/design-spec.md
# or pick a model directly:
python scripts/setup_tryon.py --project-id $PROJECT --model flash
python scripts/setup_tryon.py --project-id $PROJECT --model pro --safety-level block_some
```

## Model tiers

| Label | Model ID | Best for |
|-------|----------|----------|
| `flash` | `gemini-2.5-flash-image` | High-volume, cost-sensitive |
| `pro` | `gemini-2.5-pro-image` | Luxury, editorial |

## Skill

See [skills/retail-virtual-tryon/SKILL.md](../../skills/retail-virtual-tryon/SKILL.md) for the conversational agent guide.

## License

Apache 2.0
