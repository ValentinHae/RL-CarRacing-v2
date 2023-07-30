# RL-CarRacing-v2

This repository contains the code to train and execute an agent on the OpenAI CarRace-v2 environment.

# Installation

The repo contains a requirements.txt which can be used to install all necessary dependencies to execute the code.
You need pipenv to install all dependencies with the following command, since an old versions of tensorflow was used to utilize CUDA.

```
pipenv install
```

With the installed tensorflow version (2.5.0), in order to use CUDA you need to install CUDA 11.2 and cuDNN 8.1. Also Python 3.8 is required for execution.
You can however also run the script without CUDA.
