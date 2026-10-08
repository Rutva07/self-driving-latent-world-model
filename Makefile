.PHONY: install test smoke synthetic train evaluate docker-smoke
install:
	python -m pip install -e '.[dev]'
test:
	python -m pytest -q
smoke:
	python scripts/smoke_test.py
synthetic:
	python scripts/generate_synthetic.py --out data/synthetic --train 640 --val 128 --test 128
train:
	python scripts/train.py --config configs/train.yaml --train-dir data/prepared/train --val-dir data/prepared/val
evaluate:
	python scripts/evaluate.py --checkpoint outputs/waymo/best.pt --data-dir data/prepared/val
docker-smoke:
	docker build -t latent-world-model . && docker run --rm latent-world-model
