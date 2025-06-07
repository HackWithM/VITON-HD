# body_pose.py
import argparse
import os
from PIL import Image
import json # Ensure json is imported at the top

def fake_pose(image_path, rendered_image_output_path, keypoints_json_output_path):
    # Ensure parent directories for output files exist
    os.makedirs(os.path.dirname(rendered_image_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(keypoints_json_output_path), exist_ok=True)

    # Fake output (copy image as pose, save as PNG)
    img = Image.open(image_path)
    img.save(rendered_image_output_path) # Will save as PNG if path ends with .png

    # Create fake keypoints file with a dummy person and slightly varied pose_keypoints_2d
    # OpenPose typically has 25 keypoints, each (x, y, confidence) = 75 values
    fake_keypoints = []
    for i in range(25): # 25 keypoints
        x = 10.0 + i * 5  # Slightly varying x
        y = 15.0 + i * 3  # Slightly varying y
        confidence = 1.0
        fake_keypoints.extend([x, y, confidence])

    dummy_person_data = {
        "pose_keypoints_2d": fake_keypoints,
        "face_keypoints_2d": [], # Keep others empty for simplicity
        "hand_left_keypoints_2d": [],
        "hand_right_keypoints_2d": [],
        "pose_keypoints_3d": [],
        "face_keypoints_3d": [],
        "hand_left_keypoints_3d": [],
        "hand_right_keypoints_3d": []
    }
    keypoints_content = {
        "version": 1.3,
        "people": [dummy_person_data]
    }
    # import json # json is already imported at the top of the file
    with open(keypoints_json_output_path, 'w') as f:
        json.dump(keypoints_content, f)

    # Optional: Create a dummy densepose image if still needed by any part of the original flow
    # This part might be removable if densepose is not strictly required by test.py
    # For now, let's keep a similar structure but save it based on the new paths.
    # If densepose is needed, its path should also be an argument.
    # For simplicity, we'll omit direct densepose generation here unless confirmed necessary.
    # If densepose is needed, it would typically be in test_data/test/densepose/
    # densepose_dir = os.path.join(os.path.dirname(os.path.dirname(rendered_image_output_path)), 'densepose')
    # os.makedirs(densepose_dir, exist_ok=True)
    # base_name = os.path.basename(rendered_image_output_path).replace('_rendered.png', '.jpg')
    # img.save(os.path.join(densepose_dir, base_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fake pose and keypoints data.")
    parser.add_argument("--image_path", required=True, help="Path to the input person image.")
    parser.add_argument("--rendered_image_output_path", required=True, help="Full path to save the rendered pose image (e.g., .../openpose-img/name_rendered.png).")
    parser.add_argument("--keypoints_json_output_path", required=True, help="Full path to save the keypoints JSON file (e.g., .../openpose-json/name_keypoints.json).")
    args = parser.parse_args()

    fake_pose(args.image_path, args.rendered_image_output_path, args.keypoints_json_output_path)
