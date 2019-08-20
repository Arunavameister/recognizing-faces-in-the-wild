import glob
from PIL import Image
import os
import numpy as np



save_folder = 'single_images'


def single_images(folder):
    for family in glob.iglob(f'{folder}/*'):
        for members in glob.iglob(f'{family}/*'):
            num_files = len(glob.glob(f"{members}/*.jpg"))
            if len(os.listdir(f'{members}')) == 0:
                print(f'Empty directory : {members}')
                continue

            photos = [f for f in glob.glob(f"{members}/*.jpg")]
            index = np.random.choice(num_files, 1)
            img = Image.open(f'{photos[index.item()]}').convert('RGB')
            img = np.array(img) / 255.
            family_name = family.split('/')[1]
            member_name = members.split('/')[2]
            save_path = f'{save_folder}/{family_name}_{member_name}'
            np.save(f'{save_path}.npy', img)

if __name__ == "__main__":
    single_images('train')
