# syntax=docker/dockerfile:1
FROM 9510c66c89ef
COPY . /app

RUN make /app

ADD requirements.txt ./
RUN python3 -m pip install -r requirements.txt

ADD . ./