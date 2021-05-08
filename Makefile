.SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: debug debug-mac train-identity train-dropout train-cropout train-crop train-gausian train-jpegdrop train-jpegmask train-combined test

dataset := bit

download:
	wget http://images.cocodataset.org/zips/train2014.zip
	wget http://images.cocodataset.org/zips/test2014.zip
	unzip train2014.zip
	unzip test2014.zip
	mkdir coco2014
	mv train2014 coco2014/train
	mv test2014 coco2014/test
	mv coco2014 ${HOME}/data

debug: train_dis = jpeg_drop
debug: test_dis = jpeg
debug: gpu_ids = [0,1]
debug:
	poetry run python cmd/train.py config/experiment@experiment=debug config/distortion@train_distortion=$(train_dis) config/distortion@test_distortion=$(test_dis) training.epochs=1 training.test_interval=1 gpu_ids=$(gpu_ids) config/dataset@dataset=$(dataset)

debug-mac: train_dis = combined
debug-mac: test_dis = identity
debug-mac:
	poetry run python cmd/train.py config/experiment@experiment=debug config/distortion@train_distortion=$(train_dis) config/distortion@test_distortion=$(test_dis) training.epochs=1 training.test_interval=1 data.root=/Users/tanimu/data gpu_ids=[] config/dataset@dataset=$(dataset)

train-identity:
	poetry run python cmd/train.py \
		experiment.tags=[distortion:identity,dataset:$(dataset)] \
		experiment.use_comet=true \
		experiment.prefix=$(dataset)_identity \
		config/distortion@train_distortion=identity \
		config/distortion@test_distortion=identity \
		config/dataset@dataset=$(dataset)

train-dropout: p := 0.3
train-dropout:
	poetry run python cmd/train.py \
		experiment.tags=[distortion:dropout,dataset:$(dataset)] \
		experiment.use_comet=true \
		experiment.prefix=$(dataset)_dropout_p:$(p) \
		config/distortion@train_distortion=dropout \
		config/distortion@test_distortion=dropout \
		train_distortion.probability=$(p) \
		test_distortion.probability=$(p) \
		config/dataset@dataset=$(dataset)

train-cropout: p := 0.3
train-cropout:
	poetry run python cmd/train.py \
		experiment.tags=[distortion:cropout,dataset:$(dataset)] \
		expeciment.use_comet=true \
		experiment.prefix=$(dataset)_cropout_p:$(p) \
		config/distortion@train_distortion=cropout aa\
		config/distortion@test_distortion=cropout \
		train_distortion.probability=$(p) \
		test_distortion.probability=$(p) \
		config/dataset@dataset=$(dataset)

train-crop: p := 0.035
train-crop:
	poetry run python cmd/train.py \
		experiment.tags=[distortion:crop,dataset:$(dataset)] \
		experiment.use_comet=true \
		experiment.prefix=$(dataset)_crop_p:$(p) \
		config/distortion@train_distortion=crop \
		config/distortion@test_distortion=crop \
		train_distortion.probability=$(p) \
		test_distortion.probability=$(p) \
		config/dataset@dataset=$(dataset)

train-gaussian: sigma := 2.0
train-gaussian:
	poetry run python cmd/train.py \
		experiment.tags=[distortion:gaussian,dataset:$(dataset)] \
		experiment.use_comet=true \
		experiment.prefix=$(dataset)_gaussian_sigma:$(sigma) \
		config/distortion@train_distortion=gaussian_blur \
		config/distortion@test_distortion=gaussian_blur \
		train_distortion.sigma=$(sigma) \
		test_distortion.sigma=$(sigma) \
		config/dataset@dataset=$(dataset)

train-jpegdrop:
	poetry run python cmd/train.py \
		experiment.tags=[distortion:jpeg_drop,dataset:$(dataset)] \
		experiment.use_comet=true \
		experiment.prefix=$(dataset)_jpeg_drop \
		config/distortion@train_distortion=jpeg_drop \
		config/distortion@test_distortion=jpeg \
		config/dataset@dataset=$(dataset)

train-jpegmask:
	poetry run python cmd/train.py \
		experiment.tags=[distortion:jpeg_mask,dataset:$(dataset)] \
		experiment.use_comet=true \
		experiment.prefix=$(dataset)_jpeg_mask \
		config/distortion@train_distortion=jpeg_mask \
		config/distortion@test_distortion=jpeg \
		config/dataset@dataset=$(dataset)

train-combined: test_dis = jpeg
train-combined:
	poetry run python cmd/train.py \
		experiment.tags=[distortion:combined,dataset:$(dataset)] \
		experiment.use_comet=true \
		experiment.prefix=$(dataset)_combined \
		config/distortion@train_distortion=combined \
		config/distortion@test_distortion=$(test_dis) \
		config/dataset@dataset=$(dataset)

test: dis :=
test: train_dis :=
test: train_exp := 
test:
	poetry run python cmd/test.py \
		experiment.tags=[distortion:$(dis),dataset:$(dataset)] \
		experiment.use_comet=true \
		experiment.prefix=$(dataset)_test_$(train_dis)_for_$(dis) \
		experiment.model_path=.log/train/$(train_exp)/parameters.pth \
		config/distortion@distortion=$(dis) \
		config/dataset@dataset=$(dataset)
