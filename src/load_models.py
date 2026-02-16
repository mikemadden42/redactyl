import glob
import os
import subprocess
import sys
import tarfile
import urllib.request

import openvino as ov

# Define the script directory so we know exactly where to save the CPU model
script_dir = os.path.dirname(os.path.abspath(__file__))

# Bypass the slow connectivity check
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
from paddleocr import PaddleOCR

print("1. Loading PaddleOCR (Detection Model)...")
PaddleOCR(use_textline_orientation=False, lang="en")

print("\n2. Searching for the downloaded models...")
home_dir = os.path.expanduser("~")
paddle_models_dir = os.path.join(home_dir, ".paddlex", "official_models")

det_jsons = glob.glob(
    os.path.join(paddle_models_dir, "*det*", "**", "inference.json"), recursive=True
)

if not det_jsons:
    print("❌ Could not find inference.json. Something is really wrong.")
else:
    det_json_path = det_jsons[0]
    det_model_dir = os.path.dirname(det_json_path)

    onnx_path = os.path.join(det_model_dir, "model.onnx")

    print("\n3. Checking for ONNX conversion...")
    if not os.path.exists(onnx_path):
        print("   Translating Paddle JSON to ONNX (this only happens once)...")
        paddle2onnx_bin = os.path.join(os.path.dirname(sys.executable), "paddle2onnx")
        subprocess.run(
            [
                paddle2onnx_bin,
                "--model_dir",
                det_model_dir,
                "--model_filename",
                "inference.json",
                "--params_filename",
                "inference.pdiparams",
                "--save_file",
                onnx_path,
                "--opset_version",
                "11",
                "--enable_onnx_checker",
                "True",
            ],
            check=True,
        )
    else:
        print("   ✅ ONNX file already exists.")

    print("\n4. Loading and compiling the ONNX model to the NPU...")
    core = ov.Core()
    ov_model = core.read_model(model=onnx_path)

    # FIX: Lock the dynamic shape to a static shape for the NPU
    input_layer = ov_model.inputs[0]
    print(f"   Original shape: {input_layer.partial_shape}")

    # Reshape to [Batch, Channels, Height, Width] -> [1, 3, 960, 960]
    ov_model.reshape({input_layer.any_name: [1, 3, 960, 960]})
    print("   Reshaped to static: [1, 3, 960, 960]")

    print("   Compiling to Intel AI Boost NPU (this may take a moment)...")
    compiled_model = core.compile_model(model=ov_model, device_name="NPU")

    print(
        "\n✅ SUCCESS! The static ONNX detection model is compiled and residing in NPU memory."
    )

print("\n5. Checking for CPU Recognition Model...")
rec_model_url = (
    "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar"
)
rec_tar_path = os.path.join(script_dir, "en_PP-OCRv4_rec_infer.tar")
rec_dir_path = os.path.join(script_dir, "en_PP-OCRv4_rec_infer")

if not os.path.exists(rec_dir_path):
    print("   Downloading CPU Recognition Model...")
    urllib.request.urlretrieve(rec_model_url, rec_tar_path)

    print("   Extracting Recognition Model...")
    with tarfile.open(rec_tar_path, "r") as tar:
        tar.extractall(path=script_dir)

    os.remove(rec_tar_path)  # Clean up the tar archive
    print("✅ SUCCESS! Recognition model extracted to src/")
else:
    print("✅ CPU Recognition Model already exists in src/")
