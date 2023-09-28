build-dataset: src/dataset/preprocess.py
	echo "Downloading dataset..."
	gdown https://drive.google.com/uc?id=1cF-u9N7miok5-KUiriTdvA62edxIs1I_
	echo "Unzipping dataset..."
	unzip -qq public_coin_dataset.zip
	echo "Preprocessing dataset..."
	python src/dataset/preprocess.py
	echo "Done!"
	rm -rf public_coin_dataset.zip
	rm -rf public_coin_dataset