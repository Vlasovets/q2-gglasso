# syntax=docker/dockerfile:1
FROM quay.io/qiime2/core:2022.2
#FROM q2-gglasso-2022-2

COPY . ./
WORKDIR .
ADD requirements.txt ./
RUN python -m pip install -r requirements.txt

RUN ["apt-get", "update"]
RUN ["apt-get", "-y", "install", "vim"]
