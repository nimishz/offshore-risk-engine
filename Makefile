.PHONY: install data analysis validate docs test lint notebooks dashboard all

install:
	pip install -r requirements.txt && pip install -e .

data:
	python scripts/generate_synthetic_data.py
	python scripts/build_data_dictionary.py

lint:
	ruff check .
	ruff format --check .

test:
	pytest

analysis:
	python scripts/run_analysis.py

validate:
	python scripts/run_validation.py

docs:
	python scripts/render_docs.py

notebooks:
	python scripts/build_notebooks.py

dashboard:
	streamlit run dashboard/app.py --server.address localhost

all: data lint test analysis validate docs notebooks
