#!/bin/bash
# usage: upload.sh tokenizer|base|sft|final   (never fails the caller)
/root/nanochat/.venv/bin/python /root/upload.py "$1" >> /root/upload.log 2>&1 || echo "upload $1 failed (see upload.log)"
