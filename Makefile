# Makefile
format-black:
	@black .
format-isort:
	@isort .
lint-black:
	@black . --check
lint-isort:
	@isort . --check
lint-flake8:
	@flake8 .
lint-mypy:
	@mypy .
lint-mypy-report:
	@mypy . --html-report ./mypy-html
format: format-black format-isort
lint: lint-black lint-isort lint-flake8 lint-mypy
