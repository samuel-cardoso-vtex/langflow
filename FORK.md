
# Custom Components
src/lfx/src/lfx/components/conversations
src/lfx/src/lfx/components/litellm

# Run

## Initial setup
pyenv uninstall 3.11.13
pyenv install 3.11.13
pyenv global 3.11.13
pyenv shell 3.11.13

uv python uninstall 3.13
uv python install 3.13

## Running
make init
make lfx_build
make build_frontend
make build_component_index
uv run langflow run --log-level debug --dev

## Clean helpers
make clean_all
rm -rf dist .venv .mypy_cache src/frontend/build src/frontend/node_modules src/frontend/.dspy_cache
