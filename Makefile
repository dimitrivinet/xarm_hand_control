PORTABLE_DIR=portable

build_portable:
	mkdir -p ${PORTABLE_DIR}/modules/training ${PORTABLE_DIR}/dataset ${PORTABLE_DIR}/models
	cp src/main_processing.py ${PORTABLE_DIR}/main.py
	cp src/modules/utils.py ${PORTABLE_DIR}/modules/utils.py
	cp src/modules/training/model.py ${PORTABLE_DIR}/modules/training/model.py
	cp src/.env.default ${PORTABLE_DIR}/.env
	cp src/classes.json ${PORTABLE_DIR}/dataset/dataset.json

clean_portable:
	rm -rf ${PORTABLE_DIR}
