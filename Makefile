.SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: debug debug-mac train-identity train-dropout train-cropout train-crop train-gausian train-jpegdrop train-jpegmask

debug: train_dis = jpeg_drop
debug: test_dis = jpeg
debug: gpu_ids = [0,1]
debug:
	poetry run python cmd/train.py config/experiment@experiment=debug config/distortion@train_distortion=$(train_dis) config/distortion@test_distortion=$(test_dis) training.epochs=1 training.test_interval=1 gpu_ids=$(gpu_ids)

debug-mac: train_dis = combined
debug-mac: test_dis = identity
debug-mac:
	poetry run python cmd/train.py config/experiment@experiment=debug config/distortion@train_distortion=$(train_dis) config/distortion@test_distortion=$(test_dis) training.epochs=1 training.test_interval=1 data.root=/Users/tanimu/data gpu_ids=[]

train-identity:
	poetry run python cmd/train.py \
		experiment.tags=distortion:identity \
		experiment.use_comet=true \
		experiment.prefix=identity \
		config/distortion@train_distortion=identity \
		config/distortion@test_distortion=identity

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

train-combined: test_dis = jpeg
train-combined:
	poetry run python cmd/train.py \
		experiment.tags=distortion:combined \
		experiment.use_comet=true \
		experiment.prefix=combined \
		config/distortion@train_distortion=combined \
		config/distortion@test_distortion=$(test_dis)

test: dis :=
test: train_dis :=
test: train_exp := 
test:
	poetry run python cmd/test.py \
		experiment.tags=distortion:$(dis) \
		experiment.use_comet=true \
		experiment.prefix=test_$(train_dis)_for_$(dis) \
		experiment.model_path=.log/train/$(train_exp)/parameters.pth \
		config/distortion@distortion=$(dis)
