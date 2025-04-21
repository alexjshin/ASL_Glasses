#!/bin/bash

echo "Launch RTMP server on port 1935..."
docker run -d --name asl-rtmp -p 1935:1935 -p 8080:80 tiangolo/nginx-rtmp

echo "Starting ASL Flask server"
source ../venv/bin/activate 
python server.py 