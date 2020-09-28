#!/bin/bash

base_url="https://pjreddie.com/media/files"
for model in yolov3 yolov3-tiny; do
  wget "$base_url/$model.weights"
done
