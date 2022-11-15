pip install kaggle

git clone https://github.com/yashchks87/helper_functions_cv.git

export KAGGLE_USERNAME=yashchoksi16
export KAGGLE_KEY=f355b838448de9fb224e6fe224b8d459

kaggle competitions download -c airbus-ship-detection

mkdir files/

unzip airbus-ship-detection.zip  -d  ./files/

rm airbus-ship-detection.zip