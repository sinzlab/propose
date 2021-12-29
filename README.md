# PROPOSE

**PRO**babilistic **POSE** estimation

[![Test](https://github.com/PPierzc/propose/workflows/Test/badge.svg)](https://github.com/PPierzc/propose/actions/workflows/test.yml)
[![Black](https://github.com/PPierzc/propose/workflows/Black/badge.svg)](https://github.com/PPierzc/propose/actions/workflows/black.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

### Requirements
This project requires that you have the following installed:
- `docker`
- `docker-compose`

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
