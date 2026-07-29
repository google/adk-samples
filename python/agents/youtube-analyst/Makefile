TESTOPTS := ""
TESTPATH := "tests/"
UV_OPTS := --default-index https://pypi.org/simple

.PHONY: install
install: ## Install virtual environment with uv
	@echo "🚀 Creating virtual environment using uv"
	@UV_NO_CONFIG=1 uv sync $(UV_OPTS) --all-groups --frozen

.PHONY: check
check: ## Check lock file consistency and run static code analysis
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@UV_NO_CONFIG=1 uv lock $(UV_OPTS) --locked
	@echo "🚀 Linting code: Running ruff"
	@UV_NO_CONFIG=1 uv run $(UV_OPTS) ruff check --fix
	@echo "🚀 Static type checking: Running basedpyright"
	@UV_NO_CONFIG=1 uv run $(UV_OPTS) basedpyright youtube_analyst
	@echo "🚀 Checking for obsolete dependencies: Running deptry"
	@UV_NO_CONFIG=1 uv run $(UV_OPTS) deptry youtube_analyst

.PHONY: test
test: ## Run all tests
	@echo "🚀 Testing code: Running pytest"
	@UV_NO_CONFIG=1 uv run $(UV_OPTS) pytest $(TESTPATH) $(TESTOPTS) \
		--cov \
		--cov-config=pyproject.toml \
		--cov-report=xml \
		--cov-report=term \
		--cov-report=term-missing

.PHONY: web
web: ## Run the ADK web demo server
	@UV_NO_CONFIG=1 uv run $(UV_OPTS) adk web --reload .

.PHONY: api_server
api_server: ## Run the ADK FastAPI server
	@UV_NO_CONFIG=1 uv run $(UV_OPTS) adk api_server .

.PHONY: cli
cli: ## Run the ADK CLI interface
	@UV_NO_CONFIG=1 uv run $(UV_OPTS) adk run youtube_analyst

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
