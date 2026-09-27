.PHONY: install data analysis validate test notebooks dashboard all

install:
	pip install -r requirements.txt && pip install -e .

data:
	python scripts/generate_synthetic_data.py
	python scripts/build_data_dictionary.py

test:
	pytest

analysis:
	python scripts/run_analysis.py

validate:
	python scripts/run_validation.py

notebooks:
	python scripts/build_notebooks.py

dashboard:
	streamlit run dashboard/app.py

all: data test analysis validate notebooks
