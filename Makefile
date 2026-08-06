SOURCES := openai_embeddings_model tests

.PHONY: fmt install update-all mkdocs pytest

# Development
fmt:
	@isort $(SOURCES)
	@black $(SOURCES)
	@ruff check --fix $(SOURCES)
	@gitleaks dir --config .gitleaks.toml --no-banner --redact .

install:
	poetry install --all-extras --all-groups

update-all:
	poetry update

mkdocs:
	mkdocs serve

pytest:
	python -m pytest --cov=languru --cov-config=.coveragerc --cov-report=xml:coverage.xml
