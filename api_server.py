from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import shutil
import os
import subprocess
import uuid # For generating unique session IDs

app = FastAPI()

DATASET_DIR = "test_data"
# RESULT_DIR = "results/test_output" # This will be replaced by user-specific session directories
SESSION_BASE_RESULT_DIR = "results/user_sessions" # New base for session outputs

# Define specific target directories within DATASET_DIR/test/
TEST_SUBDIR = "test"
TEST_IMAGE_DIR = os.path.join(DATASET_DIR, TEST_SUBDIR, "image")
TEST_CLOTH_DIR = os.path.join(DATASET_DIR, TEST_SUBDIR, "cloth")
TEST_OPENPOSE_IMG_DIR = os.path.join(DATASET_DIR, TEST_SUBDIR, "openpose-img")
TEST_OPENPOSE_JSON_DIR = os.path.join(DATASET_DIR, TEST_SUBDIR, "openpose-json")
TEST_IMAGE_PARSE_DIR = os.path.join(DATASET_DIR, TEST_SUBDIR, "image-parse") # For segmentation output

PAIRS_FILE = os.path.join(DATASET_DIR, "test_pairs.txt") # Located at test_data/test_pairs.txt
# VITONHD_TEST_NAME = "vitonhd_test" # This will be dynamic per session
# FINAL_OUTPUT_DIR = os.path.join(RESULT_DIR, VITONHD_TEST_NAME) # This will be dynamic per session

# Ensure required folders exist (these are for intermediate processing files)
os.makedirs(TEST_IMAGE_DIR, exist_ok=True)
os.makedirs(TEST_CLOTH_DIR, exist_ok=True)
os.makedirs(TEST_OPENPOSE_IMG_DIR, exist_ok=True)
os.makedirs(TEST_OPENPOSE_JSON_DIR, exist_ok=True)
os.makedirs(TEST_IMAGE_PARSE_DIR, exist_ok=True)
os.makedirs(SESSION_BASE_RESULT_DIR, exist_ok=True) # Base for all session outputs

@app.get("/")
async def read_root():
    return {"message": "Welcome to the VITON-HD API!"}

@app.get("/favicon.ico", status_code=204)
async def favicon_ico():
    return None

@app.post("/tryon/")
async def try_on(person_img: UploadFile = File(...), cloth_img: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    current_session_output_dir = os.path.join(SESSION_BASE_RESULT_DIR, session_id)
    os.makedirs(current_session_output_dir, exist_ok=True) # Create unique dir for this session's final output

    person_filename = person_img.filename
    cloth_filename = cloth_img.filename

    # Save to test_data/test/image/ and test_data/test/cloth/
    person_path = os.path.join(TEST_IMAGE_DIR, person_filename)
    cloth_path = os.path.join(TEST_CLOTH_DIR, cloth_filename)

    # 1. Save images
    with open(person_path, "wb") as f:
        shutil.copyfileobj(person_img.file, f)
    with open(cloth_path, "wb") as f:
        shutil.copyfileobj(cloth_img.file, f)

    # 2. Run OpenPose via body_pose.py
    try:
        person_basename = person_filename.split(".")[0]
        rendered_image_name = f"{person_basename}_rendered.png"
        keypoints_json_name = f"{person_basename}_keypoints.json"

        rendered_image_output_path = os.path.join(TEST_OPENPOSE_IMG_DIR, rendered_image_name)
        keypoints_json_output_path = os.path.join(TEST_OPENPOSE_JSON_DIR, keypoints_json_name)

        # Enhanced subprocess call for body_pose.py
        process_result = subprocess.run(
            [
                "python", "body_pose.py",
                "--image_path", person_path,
                "--rendered_image_output_path", rendered_image_output_path,
                "--keypoints_json_output_path", keypoints_json_output_path
            ],
            check=True,
            capture_output=True,  # Capture stdout/stderr
            text=True  # Decode as text
        )
    except subprocess.CalledProcessError as e:
        error_details = {
            "detail": "OpenPose execution (body_pose.py) failed",
            "returncode": e.returncode,
            "cmd": " ".join(e.cmd) if e.cmd else "N/A",
            "stdout": e.stdout if e.stdout is not None else "No stdout captured",
            "stderr": e.stderr if e.stderr is not None else "No stderr captured (or stderr was empty)"
        }
        print(f"Error running body_pose.py: {error_details}")  # Server-side logging
        return JSONResponse(status_code=500, content=error_details)
    except FileNotFoundError as e:
       error_details = {
           "detail": "Failed to start body_pose.py process: File not found",
           "error": str(e),
           "filename": e.filename if hasattr(e, 'filename') else "N/A"
       }
       print(f"FileNotFoundError running body_pose.py: {error_details}")
       return JSONResponse(status_code=500, content=error_details)
    except Exception as e:
       error_details = {
           "detail": "An unexpected error occurred while trying to run body_pose.py",
           "error_type": type(e).__name__,
           "error_message": str(e)
       }
       print(f"Unexpected error running body_pose.py: {error_details}")
       return JSONResponse(status_code=500, content=error_details)

    # 3. Run Segmentation (human_parsing/inference.py)
    try:
        person_basename = person_filename.split(".")[0] # Already defined, re-use for clarity
        parse_output_filename = f"{person_basename}.png" # Segmentation output is .png
        parse_output_path = os.path.join(TEST_IMAGE_PARSE_DIR, parse_output_filename)

        segmentation_script_path = os.path.join("human_parsing", "inference.py")

        subprocess.run(
            [
                "python", segmentation_script_path,
                "--input_image", person_path, # Full path to the uploaded person image
                "--output_image", parse_output_path # Full path for the parse map
            ],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        error_details = {
            "detail": "Segmentation execution (human_parsing/inference.py) failed",
            "returncode": e.returncode,
            "cmd": " ".join(e.cmd) if e.cmd else "N/A",
            "stdout": e.stdout if hasattr(e, 'stdout') and e.stdout is not None else "No stdout captured",
            "stderr": e.stderr if hasattr(e, 'stderr') and e.stderr is not None else "No stderr captured"
        }
        print(f"Error running segmentation: {error_details}")
        return JSONResponse(status_code=500, content=error_details)
    except FileNotFoundError as e:
        error_details = {
            "detail": f"Failed to start segmentation process: File not found ({e.filename})",
            "error": str(e),
        }
        print(f"FileNotFoundError running segmentation: {error_details}")
        return JSONResponse(status_code=500, content=error_details)
    except Exception as e:
        error_details = {
            "detail": "An unexpected error occurred while trying to run segmentation",
            "error_type": type(e).__name__,
            "error_message": str(e)
        }
        print(f"Unexpected error running segmentation: {error_details}")
        return JSONResponse(status_code=500, content=error_details)

    # 4. Update test_pairs.txt
    with open(PAIRS_FILE, "w") as f:
        # This path structure is relative to DATASET_DIR/test/ as expected by datasets.py
        f.write(f"image/{person_filename} cloth/{cloth_filename}\n")

    # 5. Run VITON-HD model
    try:
      subprocess.run(
          [
              "python", "test.py",
              "--name", session_id, # Use unique session_id as the run name
              "--dataset_mode", "test",
              "--dataset_list", os.path.basename(PAIRS_FILE),
              "--dataset_dir", DATASET_DIR,
              "--save_dir", SESSION_BASE_RESULT_DIR, # Base directory for saving
              "--seg_checkpoint", "checkpoints/seg_final.pth",
              "--gmm_checkpoint", "checkpoints/gmm_final.pth",
              "--alias_checkpoint", "checkpoints/alias_final.pth"
          ],
          check=True,
          capture_output=True,
          text=True
      )
    except subprocess.CalledProcessError as e:
        error_details = {
            "detail": "VITON-HD model execution (test.py) failed",
            "returncode": e.returncode,
            "cmd": " ".join(e.cmd) if e.cmd else "N/A",
            "stdout": e.stdout if hasattr(e, 'stdout') and e.stdout is not None else "No stdout captured",
            "stderr": e.stderr if hasattr(e, 'stderr') and e.stderr is not None else "No stderr captured (or stderr was empty)"
        }
        print(f"Error running test.py: {error_details}")  # Server-side logging
        return JSONResponse(
            status_code=500,
            content=error_details
        )
    except FileNotFoundError as e:
        # Handle cases where 'python' or 'test.py' is not found
        error_details = {
            "detail": "Failed to start test.py process: File not found",
            "error": str(e),
            "filename": e.filename if hasattr(e, 'filename') else "N/A"
        }
        print(f"FileNotFoundError running test.py: {error_details}")
        return JSONResponse(status_code=500, content=error_details)
    except Exception as e:
        # Catch any other unexpected errors during subprocess invocation
        error_details = {
            "detail": "An unexpected error occurred while trying to run test.py",
            "error_type": type(e).__name__,
            "error_message": str(e)
        }
        print(f"Unexpected error running test.py: {error_details}")
        return JSONResponse(status_code=500, content=error_details)

    # 6. Final output path
    # test.py saves into <save_dir>/<name>/ (e.g., results/user_sessions/<session_id>/)
    # The output filename is <person_basename>_<cloth_basename>.png as per test.py logic
    
    person_basename_for_output = os.path.splitext(person_filename)[0]
    cloth_basename_for_output = os.path.splitext(cloth_filename)[0]
    expected_output_filename = f"{person_basename_for_output}_{cloth_basename_for_output}.png"
    
    output_image_path = os.path.join(current_session_output_dir, expected_output_filename)

    if not os.path.exists(output_image_path):
        error_msg = (
            f"Output image '{expected_output_filename}' not found in '{current_session_output_dir}'. "
            "The VITON-HD process (test.py) might have failed, not produced an output, "
            "or saved it with a different name."
        )
        print(f"Error: {error_msg}") # Server-side logging
        return JSONResponse(status_code=500, content={"detail": error_msg})

    return FileResponse(output_image_path, media_type="image/png", filename="result.png")
