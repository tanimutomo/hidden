.SHELL=/bin/bash

.PHONY: debug debug-mac train-identity train-dropout train-cropout train-crop train-gausian train-jpegdrop train-jpegmask

debug:
	poetry run python cmd/train.py config/experiment@experiment=debug config/distortion@train_distortion=crop config/distortion@test_distortion=dropout training.epochs=1 training.test_interval=1

debug-mac:
	poetry run python cmd/train.py config/experiment@experiment=debug config/distortion@train_distortion=crop config/distortion@test_distortion=dropout training.epochs=1 training.test_interval=1 data.root=/Users/tanimu/data gpu_ids=[]

train-identity:
	poetry run python cmd/train.py \
		experiment.tags=distortion:identity \
		experiment.use_comet=true \
		experiment.prefix=identity

train-dropout:
	poetry run python cmd/train.py \
		experiment.tags=distortion:dropout \
		experiment.use_comet=true \
		experiment.prefix=dropout_p:0.3 \
		config/distortion@train_distortion=dropout \
		config/distortion@test_distortion=dropout \
		distortion.probability=0.3

train-cropout:
	poetry run python cmd/train.py \
		experiment.tags=distortion:cropout \
		experiment.use_comet=true \
		experiment.prefix=cropout_p:0.3 \
		config/distortion@train_distortion=cropout aa\
		config/distortion@test_distortion=cropout \
		distortion.probability=0.3

train-crop:
	poetry run python cmd/train.py \
		experiment.tags=distortion:crop \
		experiment.use_comet=true \
		experiment.prefix=crop_p:0.035 \
		config/distortion@train_distortion=crop \
		config/distortion@test_distortion=crop \
		distortion.probability=0.035

train-gaussian:
	poetry run python cmd/train.py \
		experiment.tags=distortion:gaussian \
		experiment.use_comet=true \
		experiment.prefix=gaussian_sigma:2.0 \
		config/distortion@train_distortion=gaussian_blur \
		config/distortion@test_distortion=gaussian_blur \
		distortion.sigma=2.0

train-jpegdrop:
	poetry run python cmd/train.py \
		experiment.tags=distortion:jpeg_drop \
		experiment.use_comet=true \
		experiment.prefix=jpeg_drop \
		config/distortion@train_distortion=jpeg_drop \
		config/distortion@test_distortion=jpeg

train-jpegmask:
	poetry run python cmd/train.py \
		experiment.tags=distortion:jpeg_mask \
		experiment.use_comet=true \
		experiment.prefix=jpeg_mask \
		config/distortion@train_distortion=jpeg_mask \
		config/distortion@test_distortion=jpeg
