#!/bin/bash
docker run --gpus all -it --rm --name kgat-runner -v "$(pwd)":/workspace kgat-pytorch
