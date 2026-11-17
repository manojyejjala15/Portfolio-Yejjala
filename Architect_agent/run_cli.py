"""Command-line runner: python run_cli.py sample_projects/retail_forecasting.md [--mock] -o out.md"""
import argparse
import dataclasses
import sys
from pathlib import Path

from architect_agent import ArchitectAgent, Settings
from architect_agent.llm import build_client

parser = argparse.ArgumentParser()
parser.add_argument("description_file")
parser.add_argument("-o", "--output", default="architecture.md")
parser.add_argument("--mock", action="store_true", help="use the offline mock LLM")
args = parser.parse_args()

settings = Settings.from_env()
if args.mock:
    settings = dataclasses.replace(settings, use_mock=True)
path = Path(args.description_file)
result = ArchitectAgent(build_client(settings), settings).run(path.read_text(), path.stem.replace("_", " ").title())
Path(args.output).write_text(result.markdown)
print(f"approved={result.approved} score={result.review.score} rounds={result.review_rounds} "
      f"tokens={result.usage.input_tokens + result.usage.output_tokens} -> {args.output}", file=sys.stderr)
