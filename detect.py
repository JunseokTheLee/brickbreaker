# detect.py
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A demo which runs object detection on camera frames using GStreamer.
It also provides support for Object Tracker.

Run default object detection:
python3 detect.py

Choose different camera and input encoding
python3 detect.py --videosrc /dev/video1 --videofmt jpeg

Choose an Object Tracker. Example : To run sort tracker
python3 detect.py --tracker sort

TEST_DATA=../all_models

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt
"""
import argparse
import collections
import svgwrite
import common
import gstreamer
import numpy as np
import os
import re
import time
from tracker import ObjectTracker

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])


def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
        lines = (p.match(line).groups() for line in f.readlines())
        return {int(num): text.strip() for num, text in lines}


def shadow_text(dwg, x, y, text, font_size=20):
    dwg.add(dwg.text(text, insert=(x+1, y+1), fill='black', font_size=font_size))
    dwg.add(dwg.text(text, insert=(x, y), fill='white', font_size=font_size))


def generate_svg(src_size, inference_size, inference_box, objs, labels, text_lines, trdata, trackerFlag, crop_offsets):
    # Unpack the crop offsets.
    crop_left, crop_top, crop_right, crop_bottom = crop_offsets
    # Compute the size of the visible (cropped) region.
    cropped_width = src_size[0] - crop_left - crop_right
    cropped_height = src_size[1] - crop_top - crop_bottom
    # Use the cropped size for the drawing.
    dwg = svgwrite.Drawing('', size=(cropped_width, cropped_height))
    inf_w, inf_h = inference_size
    box_x, box_y, box_w, box_h = inference_box
    # Scale factors now computed using the cropped dimensions.
    scale_x, scale_y = cropped_width / box_w, cropped_height / box_h

    for y, line in enumerate(text_lines, start=1):
        shadow_text(dwg, 10, y * 20, line)
    if trackerFlag and (np.array(trdata)).size:
        for td in trdata:
            x0, y0, x1, y1, trackID = (
                td[0].item(),
                td[1].item(),
                td[2].item(),
                td[3].item(),
                td[4].item()
            )
            overlap = 0
            for ob in objs:
                dx0, dy0, dx1, dy1 = (
                    ob.bbox.xmin.item(),
                    ob.bbox.ymin.item(),
                    ob.bbox.xmax.item(),
                    ob.bbox.ymax.item()
                )
                area = (min(dx1, x1) - max(dx0, x0)) * (min(dy1, y1) - max(dy0, y0))
                if area > overlap:
                    overlap = area
                    obj = ob

            # Relative coordinates.
            x, y, w, h = x0, y0, x1 - x0, y1 - y0
            # Convert to absolute coordinates in the inference (tensor) space.
            x, y, w, h = int(x * inf_w), int(y * inf_h), int(w * inf_w), int(h * inf_h)
            # Subtract the boxing offset.
            x, y = x - box_x, y - box_y
            # Remove the crop offset so the overlay aligns with the visible area.
            x, y = x - crop_left, y - crop_top
            # Scale coordinates to the cropped display space.
            x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
            percent = int(100 * obj.score)
            label = '{}% {} ID:{}'.format(percent, labels.get(obj.id, obj.id), int(trackID))
            shadow_text(dwg, x, y - 5, label)
            dwg.add(dwg.rect(insert=(x, y), size=(w, h),
                             fill='none', stroke='red', stroke_width='2'))
    else:
        for obj in objs:
            x0, y0, x1, y1 = list(obj.bbox)
            # Relative coordinates.
            x, y, w, h = x0, y0, x1 - x0, y1 - y0
            # Convert to absolute coordinates in the inference space.
            x, y, w, h = int(x * inf_w), int(y * inf_h), int(w * inf_w), int(h * inf_h)
            # Subtract the boxing offset.
            x, y = x - box_x, y - box_y
            # Remove the crop offset.
            x, y = x - crop_left, y - crop_top
            # Scale coordinates to the cropped display space.
            x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
            percent = int(100 * obj.score)
            label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
            shadow_text(dwg, x, y - 5, label)
            dwg.add(dwg.rect(insert=(x, y), size=(w, h),
                             fill='none', stroke='blue', stroke_width='2'))
    return dwg.tostring()


class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle whose sides are vertical or horizontal.
    """
    __slots__ = ()


def get_output(interpreter, score_threshold, top_k, image_scale=1.0):
    """Returns list of detected objects."""
    boxes = common.output_tensor(interpreter, 0)
    category_ids = common.output_tensor(interpreter, 1)
    scores = common.output_tensor(interpreter, 2)

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(category_ids[i]),
            score=scores[i],
            bbox=BBox(
                xmin=np.maximum(0.0, xmin),
                ymin=np.maximum(0.0, ymin),
                xmax=np.minimum(1.0, xmax),
                ymax=np.minimum(1.0, ymax)
            )
        )
    return [make(i) for i in range(top_k) if scores[i] >= score_threshold]


def main():
    default_model_dir = '../models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir, default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    parser.add_argument('--zoom',
                        help='The initial digital zoom applied to each frame.',
                        required=False,
                        type=int,
                        default=1)
    parser.add_argument('--videosrc', help='Which video source to use. ',
                        default='/dev/video0')
    parser.add_argument('--videofmt', help='Input video format.',
                        default='raw',
                        choices=['raw', 'h264', 'jpeg'])
    parser.add_argument('--tracker', help='Name of the Object Tracker To be used.',
                        default=None,
                        choices=[None, 'sort'])
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = common.make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = load_labels(args.labels)

    w, h, _ = common.input_image_size(interpreter)
    inference_size = (w, h)
    # Average FPS over last 30 frames.
    fps_counter = common.avg_fps_counter(30)

    def user_callback(input_tensor, src_size, inference_box, mot_tracker, crop_offsets):
        nonlocal fps_counter
        start_time = time.monotonic()
        common.set_input(interpreter, input_tensor, args.zoom)
        interpreter.invoke()
        objs = get_output(interpreter, args.threshold, args.top_k)
        end_time = time.monotonic()
        detections = []
        for n in range(len(objs)):
            element = []
            element.append(objs[n].bbox.xmin)
            element.append(objs[n].bbox.ymin)
            element.append(objs[n].bbox.xmax)
            element.append(objs[n].bbox.ymax)
            element.append(objs[n].score)
            detections.append(element)
        detections = np.array(detections)
        trdata = []
        trackerFlag = False
        if detections.any():
            if mot_tracker is not None:
                trdata = mot_tracker.update(detections)
                trackerFlag = True
            text_lines = [
                'Inference: {:.2f} ms'.format((end_time - start_time) * 1000),
                'FPS: {} fps'.format(round(next(fps_counter))),
            ]
        if len(objs) != 0:
            return generate_svg(src_size, inference_size, inference_box, objs, labels, text_lines, trdata, trackerFlag, crop_offsets)

    result = gstreamer.run_pipeline(user_callback,
                                    src_size=(1080, 960),
                                    appsink_size=inference_size,
                                    trackerName=args.tracker,
                                    videosrc=args.videosrc,
                                    videofmt=args.videofmt)


if __name__ == '__main__':
    main()
