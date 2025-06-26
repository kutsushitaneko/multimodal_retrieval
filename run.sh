#!/bin/bash

PYTHONUNBUFFERED=1 uv run multimodal_retriever.py > output.log 2>&1 &