#!/bin/bash

# for local
# first run index_worker.py in new terminal
uvicorn main:app --host 0.0.0.0 --port 10000
