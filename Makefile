.SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: debug debug-mac train-identity train-dropout train-cropout train-crop train-gausian train-jpegdrop train-jpegmask train-combined test resume

dataset := bit
gpu_ids := [0,1]
suffix := default

download: ## download dataset
	wget http://images.cocodataset.org/zips/train2014.zip
	wget http://images.cocodataset.org/zips/test2014.zip
	unzip train2014.zip
	unzip test2014.zip
	mkdir coco2014
	mv train2014 coco2014/train
	mv test2014 coco2014/test
	mv coco2014 ${HOME}/data

dataset:
	poetry run python cmd/dataset.py

debug: train_dis = jpeg_drop
debug: test_dis = jpeg
debug: ## debug in Linux with cuda
	poetry run python cmd/train.py config/experiment@experiment=debug config/distortion@train_distortion=$(train_dis) config/distortion@test_distortion=$(test_dis) training.epochs=1 training.test_interval=1 gpu_ids=$(gpu_ids) config/dataset@dataset=$(dataset)

debug-mac: train_dis = combined
debug-mac: test_dis = identity
debug-mac: ## debug in macOS
	poetry run python cmd/train.py config/experiment@experiment=debug config/distortion@train_distortion=$(train_dis) config/distortion@test_distortion=$(test_dis) training.epochs=1 training.test_interval=1 data.root=/Users/tanimu/data gpu_ids=[] config/dataset@dataset=$(dataset)

train-identity: ## train the model with identity
	poetry run python cmd/train.py \
		experiment.tags=[distortion:identity,dataset:$(dataset)] \
		experiment.use_comet=true \
		experiment.prefix=$(dataset)_identity_$(suffix) \
		config/distortion@train_distortion=identity \
		config/distortion@test_distortion=identity \
		config/dataset@dataset=$(dataset) \
		gpu_ids=$(gpu_ids)

train-dropout: p := 0.3
train-dropout: ## train the model with dropout
	poetry run python cmd/train.py \
		experiment.tags=[distortion:dropout,dataset:$(dataset)] \
		experiment.use_comet=true \
		experiment.prefix=$(dataset)_dropout_p:$(p)_$(suffix) \
		config/distortion@train_distortion=dropout \
		config/distortion@test_distortion=dropout \
		train_distortion.probability=$(p) \
		test_distortion.probability=$(p) \
		config/dataset@dataset=$(dataset) \
		gpu_ids=$(gpu_ids)

train-cropout: p := 0.3
train-cropout: ## train the model with cropout
	poetry run python cmd/train.py \
		experiment.tags=[distortion:cropout,dataset:$(dataset)] \
		experiment.use_comet=true \
		experiment.prefix=$(dataset)_cropout_p:$(p) \
		config/distortion@train_distortion=cropout \
		config/distortion@test_distortion=cropout \
		train_distortion.probability=$(p) \
		test_distortion.probability=$(p) \
		config/dataset@dataset=$(dataset) \
		gpu_ids=$(gpu_ids)

train-crop: p := 0.035
train-crop: ## train the model with crop
	poetry run python cmd/train.py \
		experiment.tags=[distortion:crop,dataset:$(dataset)] \
		experiment.use_comet=true \
		experiment.prefix=$(dataset)_crop_p:$(p) \
		config/distortion@train_distortion=crop \
		config/distortion@test_distortion=crop \
		train_distortion.probability=$(p) \
		test_distortion.probability=$(p) \
		config/dataset@dataset=$(dataset) \
		gpu_ids=$(gpu_ids)

train-gaussian: sigma := 2.0
train-gaussian: ## train the model with gaussian
	poetry run python cmd/train.py \
		experiment.tags=[distortion:gaussian,dataset:$(dataset)] \
		experiment.use_comet=true \
		experiment.prefix=$(dataset)_gaussian_sigma:$(sigma)_$(suffix) \
		config/distortion@train_distortion=gaussian_blur \
		config/distortion@test_distortion=gaussian_blur \
		train_distortion.sigma=$(sigma) \
		test_distortion.sigma=$(sigma) \
		config/dataset@dataset=$(dataset) \
		gpu_ids=$(gpu_ids)

train-jpegdrop: ## train the model with jpegdrop
	poetry run python cmd/train.py \
		experiment.tags=[distortion:jpeg_drop,dataset:$(dataset)] \
		experiment.use_comet=true \
		experiment.prefix=$(dataset)_jpeg_drop_$(suffix) \
		config/distortion@train_distortion=jpeg_drop \
		config/distortion@test_distortion=jpeg \
		config/dataset@dataset=$(dataset) \
		gpu_ids=$(gpu_ids)

train-jpegmask: ## train the model with jpetmask
	poetry run python cmd/train.py \
		experiment.tags=[distortion:jpeg_mask,dataset:$(dataset)] \
		experiment.use_comet=true \
		experiment.prefix=$(dataset)_jpeg_mask_$(suffix) \
		config/distortion@train_distortion=jpeg_mask \
		config/distortion@test_distortion=jpeg \
		config/dataset@dataset=$(dataset) \
		gpu_ids=$(gpu_ids)

train-combined: test_dis = jpeg
train-combined: ## train the model with combined
	poetry run python cmd/train.py \
		experiment.tags=[distortion:combined,dataset:$(dataset)] \
		experiment.use_comet=true \
		experiment.prefix=$(dataset)_combined \
		config/distortion@train_distortion=combined \
		config/distortion@test_distortion=$(test_dis) \
		config/dataset@dataset=$(dataset) \
		gpu_ids=$(gpu_ids)

test: dis :=
test: train_dis :=
test: train_exp := 
test: ## test the model
	poetry run python cmd/test.py \
		experiment.tags=[distortion:$(dis),dataset:$(dataset)] \
		experiment.use_comet=true \
		experiment.prefix=$(dataset)_test_$(train_dis)_for_$(dis)_$(suffix) \
		experiment.model_path=.log/train/$(train_exp)/parameters.pth \
		config/distortion@distortion=$(dis) \
		config/dataset@dataset=$(dataset) \
		gpu_ids=$(gpu_ids)

resume: name :=
resume: comet_key :=
resume: train_dis :=
resume: test_dis :=
resume: ## resume training
	poetry run python cmd/train.py \
		config/experiment@experiment=resume \
		experiment.use_comet=true \
		experiment.name=$(name) \
		experiment.comet_key=$(comet_key) \
		config/distortion@train_distortion=$(train_dis) \
		config/distortion@test_distortion=$(test_dis) \
		config/dataset@dataset=$(dataset) \
		gpu_ids=$(gpu_ids)

help: ## display this help screen
		@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[$$()% a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
