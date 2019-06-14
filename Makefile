test:
	python -m pytest tests/


conda:
	conda build --output-folder dist conda.recipe


clean:
	rm -rf .coverage build/ dist/* *.html
