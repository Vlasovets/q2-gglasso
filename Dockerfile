# Install the latest version of QIIME2
ARG QIIME2_RELEASE=2023.2
FROM quay.io/qiime2/core:$QIIME2_RELEASE

COPY . ./
WORKDIR .
ADD requirements.txt ./
RUN python -m pip install -r requirements.txt # using fda branch instead of master in requirements

RUN ["apt-get", "update"]
RUN ["apt-get", "-y", "install", "vim"]
