.PHONY: all data train_dense train_aria eval plots clean test

all: data train_dense train_aria eval plots

data:
	uv run python data/prepare.py

train_dense:
	uv run clew train --model dense

train_aria:
	uv run clew train --model aria --batch-size 8

eval: eval_memory eval_needle

eval_memory:
	uv run python -m eval.memory_scan

eval_needle:
	uv run python -m eval.needle_haystack

plots:
	uv run python -m eval.plot_loss

test:
	uv run pytest tests/ -q

clean:
	rm -rf runs/*/ plots/*.png
