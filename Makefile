.PHONY: install test lint run pipeline export clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

lint:
	flake8 . --max-line-length=120 --exclude=venv,.venv

run:
	streamlit run niv_streamlit_app.py

pipeline:
	python data_pipeline.py --skip-etl

pipeline-full:
	python data_pipeline.py

export:
	python powerbi_export.py

clean:
	rm -rf __pycache__ .pytest_cache tests/__pycache__
	rm -rf data/powerbi_exports data/logs
