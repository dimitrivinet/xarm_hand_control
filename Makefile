MODELS_PATH=https://github.com/dimitrivinet/xarm_hand_control/releases/download/v1.0/models.zip

get_models: download_models install_models clean_models

download_models:
	wget ${MODELS_PATH}

install_models:
	unzip models.zip -d models
	mv models ${PORTABLE_DIR}/
	rm models.zip

clean_models:
	- rm models.zip
