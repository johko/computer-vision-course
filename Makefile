.PHONY: quality

# Check code formatting
quality:
	python utils/code_formatter.py --check_only