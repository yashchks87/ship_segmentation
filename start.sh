pip install kaggle

cd scripts

git clone https://github.com/yashchks87/helper_functions_cv.git

cd ../../

kaggle competitions download -c airbus-ship-detection

mkdir files/

unzip airbus-ship-detection.zip  -d  ./files/

rm airbus-ship-detection.zip
