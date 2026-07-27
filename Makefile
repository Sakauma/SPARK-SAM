PYTHON ?= python
DATASET ?= irstd

.PHONY: test compile preflight

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

compile:
	$(PYTHON) -m compileall -q src scripts tests

preflight:
	PYTHONPATH=src $(PYTHON) scripts/train_sparksam.py \
		--config configs/training/$(DATASET)/joint_adaptation.yaml \
		--preflight-only
