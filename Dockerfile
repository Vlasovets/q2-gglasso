# syntax=docker/dockerfile:1
FROM quay.io/qiime2/core:2021.8

COPY . ./
WORKDIR .
ADD requirements.txt ./
RUN python -m pip install -r requirements.txt
