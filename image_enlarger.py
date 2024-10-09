import argparse
import os
from PIL import Image

def is_image(file):
    image_ext = ['.DNG', '.dng','.jpg', '.jpeg', '.png', '.gif', '.tiff', '.webp','.bmp' ]  
    return any(file.lower().endswith(ext) for ext in image_ext)


def enhance_image(real_image_path, produced_image_path, required_bytes):

    with open(real_image_path, 'rb') as f:
        image_bytes = f.read()

    # number of reps needed for reaching target size
    reps = required_bytes // len(image_bytes)

    # repeat the original image bytes for enlarging
    produced_image_bytes = image_bytes * reps

    # add necessary remaining bytes to reach required bytes
    rem_bytes = required_bytes % len(produced_image_bytes)
    produced_image_bytes += image_bytes[:rem_bytes]

    with open(produced_image_path, 'wb') as f:
        f.write(produced_image_bytes)

def process_images(original_folder, produced_folder, required_bytes):
    if not os.path.exists(produced_folder):
        os.makedirs(produced_folder)

    for filename in os.listdir(original_folder):
        if is_image(filename):
          real_image_path = os.path.join(original_folder, filename)
          produced_image_path = os.path.join(produced_folder, filename)

          enhance_image(real_image_path, produced_image_path ,required_bytes)
        else:
          continue

# Example usage
#python image_enlarger.py --original_folder_path path/to/original_image_folder --produced_folder_path /path/to/produced_image_folder --required_bytes 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="enlarge an image to specific size in mb.")
    parser.add_argument("--original_folder_path", type=str, help="path to the folder with original images.")
    parser.add_argument("--produced_folder_path", type=str, help="path to the folder with produced images.")
    parser.add_argument("--required_bytes", type=int, help="required size of the images in megabytes.")

    args = parser.parse_args()

    if not args.original_folder_path or not args.produced_folder_path or not args.required_bytes:
        parser.print_help()
        exit(1)

    args.required_bytes = args.required_bytes * 1024 * 1024 #100mb
    process_images(args.original_folder_path, args.produced_folder_path, args.required_bytes)
