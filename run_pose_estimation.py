from mmpose.apis import init_model, inference_topdown
from mmpose.structures import merge_data_samples
from mmdet.apis import init_detector, inference_detector
from PIL import Image
import cv2
import json
import os
import numpy as np

# 1. Load person detection model
det_model = init_detector(
    'https://download.openmmlab.com/mmdetection/v2.0/yolov3/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco.py',
    'https://download.openmmlab.com/mmdetection/v2.0/yolov3/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco.pth',
    device='cpu'
)

# 2. Load pose model
pose_model = init_model(
    'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
    'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192.py',
    device='cpu'
)

# 3. Input image
image_path = 'test_data/image/person.jpg'
image = cv2.imread(image_path)

# 4. Detect persons
det_result = inference_detector(det_model, image)
person_result = det_result[0]
bboxes = [bbox for bbox in person_result if bbox[4] > 0.5]

# 5. Run pose estimation
pose_results = inference_topdown(pose_model, image, bboxes)
pose_output = merge_data_samples(pose_results)

# 6. Draw poses
pose_img = pose_model.visualize(
    image, pose_output, draw_bbox=False, show=False
)

# Ensure output directories exist
output_img_dir = 'test_data/openpose-img'
output_json_dir = 'test_data/openpose-json'
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_json_dir, exist_ok=True)

# 7. Save as person_rendered.png
output_img_path = os.path.join(output_img_dir, 'person_rendered.png')
cv2.imwrite(output_img_path, pose_img)
print(f"Saved pose image at {output_img_path}")

# 8. Save keypoints as person_keypoints.json
output_json_path = os.path.join(output_json_dir, 'person_keypoints.json')
if pose_output.pred_instances.keypoints.shape[0] > 0:
    # Assuming we take the first detected person's keypoints
    keypoints = pose_output.pred_instances.keypoints[0].cpu().numpy()
    keypoint_scores = pose_output.pred_instances.keypoint_scores[0].cpu().numpy()

    keypoints_2d = []
    for kp_idx in range(keypoints.shape[0]):
        keypoints_2d.extend([float(keypoints[kp_idx, 0]), float(keypoints[kp_idx, 1]), float(keypoint_scores[kp_idx])])
    
    pose_data_for_json = {
        "version": 1.3,
        "people": [
            {
                "person_id": [-1],
                "pose_keypoints_2d": keypoints_2d,
                # Add other fields if necessary, like face_keypoints_2d, hand_left_keypoints_2d, etc.
                # For now, keeping it minimal to match what datasets.py seems to parse.
            }
        ]
    }

    with open(output_json_path, 'w') as f:
        json.dump(pose_data_for_json, f, indent=4)
    print(f"Saved pose keypoints at {output_json_path}")
else:
    print("No person detected, so no keypoints JSON saved.")
