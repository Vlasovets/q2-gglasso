# QIIME 2 base image.
#
# NOTE: the `amplicon` distribution was renamed to `qiime2` in release 2026.4,
# so `quay.io/qiime2/amplicon` no longer exists for 2026.x tags.
ARG QIIME2_RELEASE=2026.7
FROM quay.io/qiime2/qiime2:${QIIME2_RELEASE}

# Set working directory
WORKDIR /code

# Install plugin dependencies first so the layer caches independently of source.
ADD requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy all plugin files into the container and install the plugin itself
COPY . ./
RUN python -m pip install --no-cache-dir --no-deps -e . \
    && qiime dev refresh-cache
