# PROPOSE

**PRO**babilistic **POSE** estimation

[![Test](https://github.com/PPierzc/propose/workflows/Test/badge.svg)](https://github.com/PPierzc/propose/actions/workflows/test.yml)
[![Black](https://github.com/PPierzc/propose/workflows/Black/badge.svg)](https://github.com/PPierzc/propose/actions/workflows/black.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/PPierzc/propose/branch/main/graph/badge.svg?token=PYI1Z06426)](https://codecov.io/gh/PPierzc/propose)

### Requirements
This project requires that you have the following installed:
- `docker`
- `docker-compose`

Ensure that you have the base image pulled from the Docker Hub.
You can get the base image by running the following command:
```
docker pull sinzlab/pytorch:v3.9-torch1.9.0-cuda11.1-dj0.12.7
```

### Installation
1. Clone the repository.
2. Navigate to the project directory.
3. Run```docker-compose run -d -p 10101:8888 notebook_server```
4. You can now open JupyterLab in your browser at [`http://localhost:10101`](http://localhost:10101).

### Run Tests

To run the tests call from the root directory

```
docker-compose run pytest tests
```

## Data
### Rat7m
You can download the Rat 7M dataset from [here](https://figshare.com/collections/Rat_7M/5295370).

### Human3.6M dataset
Due to license restrictions, the dataset is not included in the repository.
You can download it from the official [website](http://vision.imar.ro/human3.6m).

