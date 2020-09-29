# Multithread multicam real-time YOLOv3 object detection demo

This demo 

To display usage/help information, run `python demo.py -h`.

# Examples

```
# Clone pytorch-yolov3 repository and install package for YOLO object detection.
git clone https://github.com/nrsyed/pytorch-yolov3.git
cd pytorch-yolov3
pip install .
cd ..

# Clone this repository and install package.
git clone https://github.com/nrsyed/multithread-multicam
cd multithread-multicam
pip install .

# Download yolov3 and yolov3-tiny weights.
cd demo
./get_yolo_weights.sh

# Run demo using the first webcam connected to the PC (video device id 0).
python demo.py

# Run demo using all webcams connected to the PC (this checks each device at
# /dev/video* to determine if it represents a video capture device).
python demo.py -a

# Run demo with all webcams and duplicate each webcam stream 3 times. This
# simulates the effect of processing and displaying 4 times the actual
# number of webcam streams.
python demo.py -a --duplicate 3

# Run demo with all webcams using yolov3 Darknet model instead of
# yolov3-tiny and display FPS.
python demo.py -a -c yolov3.cfg -w yolov3.weights --fps

# Run demo with all webcams using yolov3, duplicate each webcam stream twice,
# display FPS, and write the result to out.mp4.
python demo.py -a -c yolov3.cfg -w yolov3.weights --duplicate 2 --fps -o out.mp4
