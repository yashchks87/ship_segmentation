# Difference between RUN and CMD?
# RUN is usually run at build step. And it commits the image with best image and make sure that
# image is having that state when it runs. Also make sure that there are multiple RUNs while CMD 
# will override any CMD comes before the last one.

FROM python:3.8.11-bullseye

COPY ./tensorflow_req.txt ./
COPY ./test_script.py ./

RUN pip install --upgrade pip

RUN pip install -r ./tensorflow_req.txt

RUN python ./test_script.py

