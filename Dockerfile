# Install the latest version of QIIME2
ARG QIIME2_RELEASE=2025.4
FROM quay.io/qiime2/amplicon:${QIIME2_RELEASE}

# Set working directory
WORKDIR /code

# Copy all plugin files into the container
COPY . ./

# Install plugin dependencies
ADD requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# Optional: install additional tools
RUN apt-get update && apt-get install -y vim
