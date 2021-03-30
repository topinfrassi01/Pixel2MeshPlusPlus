import os
import sys
import shutil
from pathlib import Path

import progressbar
import numpy as np

def main():
    n_samples = int(sys.argv[1])
    chosen_class = str(sys.argv[2])
    
    images_path = Path(os.environ["P2MPP_DIR"]) / 'data/ShapeNet/Images' / chosen_class
    objects = os.listdir(images_path)
    num_objects = len(objects)
    images_count = num_objects * 24 #there are 24 images per object
    print("Found {0} images, picking {1}...".format(images_count, n_samples))
    picks = np.random.randint(low=0, high=images_count, size=n_samples)

    img_destination = Path(os.environ["P2MPP_DIR"]) / 'data/pose_estimation/images'
    pose_destination = Path(os.environ["P2MPP_DIR"]) / 'data/pose_estimation/poses.txt'

    poses_stream = open(pose_destination, 'w+')
    try:
        print("Copying files...")
        with progressbar.ProgressBar(max_value=n_samples) as bar:
            for progress_i, p in enumerate(picks):
                object_idx = p // 24
                img_idx = p % 24

                obj = objects[object_idx]
                object_dir = images_path / obj / 'rendering'

                images_names = os.listdir(object_dir)
                
                picked_img_name = images_names[img_idx]

                picked_img_path = object_dir / picked_img_name
                
                with open(str(object_dir / 'rendering_metadata.txt'), 'r') as fs:
                    for i, line in enumerate(fs):
                        if i == img_idx:
                            metadata = line.strip()

                if metadata is None:
                    raise ValueError("Metadata not found, problem.")

                new_img_name = "_".join([obj, picked_img_name])
                new_img_path = img_destination / new_img_name
                shutil.copyfile(picked_img_path, new_img_path)
                poses_stream.write(",".join([metadata, str(new_img_name)+'\n']))
                bar.update(progress_i)
    finally:
        poses_stream.close()

if __name__ == "__main__":
    main()