test:
	python -m pytest tests/


wheel:
	python setup.py bdist_wheel sdist


conda:
	conda build --output-folder dist conda.recipe


clean:
	rm -rf .coverage build/ dist/* *.html
