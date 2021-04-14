.SHELL=/bin/bash

.PHONY: debug

debug:
	poetry run python cmd/train.py config/experiment@experiment=debug config/distortion@train_distortion=crop config/distortion@test_distortion=dropout training.epochs=1 training.test_interval=1

debug-mac:
	poetry run python cmd/train.py config/experiment@experiment=debug config/distortion@train_distortion=crop config/distortion@test_distortion=dropout training.epochs=1 training.test_interval=1 data.root=/Users/tanimu/data gpu_ids=[]
