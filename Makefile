PORTABLE_DIR=portable
MODELS_PATH=https://github.com/dimitrivinet/xarm_hand_control/releases/download/v1.0/models.zip

portable: download_models build_portable install_models

all: portable run

clean: clean_portable clean_models

run:
	python3 ${PORTABLE_DIR}/main.py

build_portable:
	mkdir -p ${PORTABLE_DIR}/modules/training ${PORTABLE_DIR}/dataset ${PORTABLE_DIR}/models
	cp src/main_processing.py ${PORTABLE_DIR}/main.py
	cp src/modules/utils.py ${PORTABLE_DIR}/modules/utils.py
	cp src/modules/training/model.py ${PORTABLE_DIR}/modules/training/model.py
	cp src/.env.default_portable ${PORTABLE_DIR}/.env
	cp src/classes.json ${PORTABLE_DIR}/dataset/dataset.json

clean_portable:
	- rm -rf ${PORTABLE_DIR}

download_models:
	wget ${MODELS_PATH}

install_models:
	unzip models.zip -d models
	mv models ${PORTABLE_DIR}/
	rm models.zip

clean_models:
	- rm models.zip
