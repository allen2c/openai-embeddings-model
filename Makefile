# Development
fmt:
	@isort openai_embeddings_model tests
	@black openai_embeddings_model tests

install:
	poetry install --all-extras --all-groups

update-all:
	poetry update

mkdocs:
	mkdocs serve

pytest:
	python -m pytest --cov=languru --cov-config=.coveragerc --cov-report=xml:coverage.xml
