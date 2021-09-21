# syntax=docker/dockerfile:1
FROM sha256:9510c66c89efdbd0b06192b16a5ebeebcea99df6fde1ea3878e1be4a26a9f403
COPY . /app

ADD requirements.txt ./
RUN python3 -m pip install -r requirements.txt
