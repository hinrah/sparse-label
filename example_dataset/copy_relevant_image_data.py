import argparse
import os
from glob import glob
import shutil

CASES_WITH_AORTA = ['s0019', 's0059', 's0096', 's0126', 's0128', 's0139', 's0144', 's0147', 's0153', 's0167', 's0168', 's0175', 's0216', 's0267', 's0286',
                    's0416', 's0447', 's0469', 's0478', 's0479', 's0481', 's0485', 's0498', 's0504', 's0505', 's0515', 's0517', 's0518', 's0520', 's0544',
                    's0547', 's0558', 's0568', 's0583', 's0589', 's0613']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help="[REQUIRED] input path of the extracted total_segmentator_dataset")
    args = parser.parse_args()

    images_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset001_totalSegmentatorAorta", "imagesTr")
    os.makedirs(images_folder, exist_ok=True)

    for case in CASES_WITH_AORTA:
        image_path = list(glob(os.path.join(args.i, "**", case, "mri.nii.gz"), recursive=True))[0]
        print(len(list(glob(os.path.join(args.i, "**", case, "mri.nii.gz"), recursive=True))))
        shutil.copy(image_path, os.path.join(images_folder, f"{case}_0000.nii.gz"))


if __name__ == "__main__":
    main()
