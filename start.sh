pip install kaggle

cd scripts

git clone https://github.com/yashchks87/helper_functions_cv.git

export KAGGLE_USERNAME=yashchoksi16
export KAGGLE_KEY=961ee48f863626a69b28671a84d21e7a

cd ../../

kaggle competitions download -c airbus-ship-detection

mkdir files/

unzip airbus-ship-detection.zip  -d  ./files/

rm airbus-ship-detection.zip
