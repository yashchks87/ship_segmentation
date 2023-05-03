pip install kaggle

cd scripts

git clone https://github.com/yashchks87/helper_functions_cv.git

export KAGGLE_USERNAME=yashchoksi16
export KAGGLE_KEY=5cdc24fa6c8455b77f63fbbae4c4179c

cd ../../

kaggle competitions download -c airbus-ship-detection

mkdir files/

unzip airbus-ship-detection.zip  -d  ./files/

rm airbus-ship-detection.zip
