import os.path as osp
import os
import numpy as np
import requests
import glob as gb
import argparse
import cv2
from PIL import Image
import subprocess
from diffmot import DiffMOT
import argparse
import yaml
from easydict import EasyDict

def extract_frames(video_path, output_folder, target_width=640):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    if video_path.endswith(".gif"):
        with Image.open(video_path) as img:
            while True:
                try:
                    frame_count = img.tell()
                    frame_img = img.convert("RGB")
                    frame_img.save(os.path.join(output_folder, f"{frame_count+1:05d}.jpg"))
                    img.seek(frame_count + 1)
                except EOFError:
                    break
    else:
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Get original video properties
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate the scale factor
        scale_factor = target_width / original_width
        target_height = int(original_height * scale_factor)

        # Variable to keep track of frame count
        frame_count = 0

        # Read frames until there are no more
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Save the frame as an image
            frame_path = os.path.join(output_folder, f"{frame_count+1:05d}.jpg")

            # Resize the frame
            resized_frame = cv2.resize(frame, (target_width, target_height))
            cv2.imwrite(frame_path, resized_frame)

            frame_count += 1

        # Release the video capture object
        cap.release()


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

def download_weights(weight_root="weights", file_name="", url=""):
    mkdirs(weight_root)

    # Specify the local file path where you want to save the downloaded file
    local_file_path = os.path.join(weight_root, file_name)

    # Check if the file already exists
    if os.path.exists(local_file_path):
        print(f"> File '{local_file_path}' already exists. Skipping download.")
        return

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Get the content of the response (file data)
        file_content = response.content

        # Specify the local file path where you want to save the downloaded file
        local_file_path = os.path.join(weight_root, file_name)

        # Write the file content to the local file
        with open(local_file_path, 'wb') as f:
            f.write(file_content)

        print(f"> File downloaded successfully to '{local_file_path}'")
    else:
        print(f"> Failed to download file: {response.status_code} - {response.reason}")

def main(args):
    with open(args.config) as f:
       config = yaml.safe_load(f)

    for k, v in vars(args).items():
       config[k] = v
    config["exp_name"] = args.config.split("/")[-1].split(".")[0]
    config["dataset"] = args.dataset
    config = EasyDict(config)
    agent = DiffMOT(config)

    if config["eval_mode"]:
        agent.eval()
    else:
        agent.train()


"""
Note that do not change the path policy
python demo.py --video_path siren.mp4 --output_video_path outputs/siren/img1 --weight DanceTrack_yolox_s.tar --yolox_type yolox-s \
        --output_txt_path outputs --config ./configs/demo_test.yaml --dataset DanceTrack
python demo.py --video_path weapon.mp4 --output_video_path outputs/weapon/img1 --weight MOT20_yolox_x.tar --yolox_type yolox-x \
        --output_txt_path outputs --config ./configs/demo_test.yaml --dataset MOT20
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help="restore checkpoint")
    parser.add_argument('--output_video_path', type=str, help="restore checkpoint")
    parser.add_argument('--output_txt_path', type=str, help="restore checkpoint")

    parser.add_argument('--weight', type=str, help="restore checkpoint")
    parser.add_argument('--yolox_type', type=str, help="restore checkpoint")

    parser.add_argument('--config', type=str, help="restore checkpoint")
    parser.add_argument('--dataset', type=str, help="restore checkpoint")

    args = parser.parse_args()

    yolo_weight_map = {
        "DanceTrack_yolox_s.tar" : "https://github.com/Kroery/DiffMOT/releases/download/v1.0/DanceTrack_yolox_s.tar",
        "DanceTrack_yolox_m.tar" : "https://github.com/Kroery/DiffMOT/releases/download/v1.0/DanceTrack_yolox_m.tar",
        "DanceTrack_yolox_l.tar" : "https://github.com/Kroery/DiffMOT/releases/download/v1.0/DanceTrack_yolox_l.tar",
        "DanceTrack_yolox_x.tar" : "https://github.com/Kroery/DiffMOT/releases/download/v1.0/DanceTrack_yolox_x.tar",
        "MOT17_yolox_x.tar" : "https://github.com/Kroery/DiffMOT/releases/download/v1.0/MOT17_yolox_x.tar",
        "MOT20_yolox_x.tar" : "https://github.com/Kroery/DiffMOT/releases/download/v1.0/MOT20_yolox_x.tar",
        "SportsMOT_yolox_x.tar" : "https://github.com/Kroery/DiffMOT/releases/download/v1.0/SportsMOT_yolox_x.tar",
        "SportsMOT_yolox_x_mix.tar" : "https://github.com/Kroery/DiffMOT/releases/download/v1.0/SportsMOT_yolox_x_mix.tar"
    }    
    
    extract_frames(video_path=args.video_path, output_folder=args.output_video_path)

    # Download re_id
    download_weights(weight_root="external/weights", file_name="dance_sbs_S50.pth", url="https://github.com/Kroery/DiffMOT/releases/download/v1.0/dance_sbs_S50.pth")
    download_weights(weight_root="external/weights", file_name="mot20_sbs_S50.pth", url="https://github.com/Kroery/DiffMOT/releases/download/v1.0/mot20_sbs_S50.pth")

    # Download DiffMOT
    DiffMOT_prefix = "MOT" if "MOT" in args.dataset else args.dataset
    download_weights(weight_root="experiments/mot_ddm_1000_deeper", file_name=f"{args.dataset}_epoch800.pt", url=f"https://github.com/Kroery/DiffMOT/releases/download/v1.0/{DiffMOT_prefix}_epoch800.pt")

    # Download YOLOX
    download_weights(file_name=args.weight, url=yolo_weight_map[args.weight])

    # Define the command to run the Python script
    command = ["python", "external/YOLOX/tools/demo.py", "image", 
               "-n", args.yolox_type, 
               "-c", f"weights/{args.weight}", 
               "--path", args.output_video_path, 
               "--output_txt_path", args.output_txt_path,
               "--conf", "0.45", "--nms", "0.45", "--tsize", "640", "--num_classes", "1", 
               "--legacy", "--save_result", "--save_txt", "--device", "gpu"]

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if the command was successful
    if result.returncode == 0:
        print("Python script executed successfully!")
    else:
        print("Error running Python script:")
        print(result.stderr)

    main(args=args)