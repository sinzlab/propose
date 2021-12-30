ARG BASE_IMAGE=sinzlab/pytorch:v3.9-torch1.9.0-cuda11.1-dj0.12.7

ARG TORCH_VERSION="torch-1.9.0"
ARG CUDA_VERSION="cu111"

# Perform multistage build to pull private repo without leaving behind
# private information (e.g. SSH key, Git token)
FROM ${BASE_IMAGE} as base
ARG DEV_SOURCE=sinzlab
ARG GITHUB_USER
ARG GITHUB_TOKEN

WORKDIR /src
# Use git credential-store to specify username and pass to use for pulling repo
RUN git config --global credential.helper store &&\
    echo https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com >> ~/.git-credentials

FROM ${BASE_IMAGE}
#COPY --from=base /src /src
ADD . /src/propose

RUN pip install -e /src/propose

RUN pip install -r /src/propose/requirements.txt

RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
RUN pip install torch-sparse -f https://data.pyg.org/whl/torch-1.9.0+cu111.html

WORKDIR /

RUN pwd