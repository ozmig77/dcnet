import argparse
import os
from PIL import Image
from glob import glob

from joblib import Parallel, delayed
import multiprocessing

def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            try:
                with Image.open(f) as img:
                    img = resize_image(img, size)
                    img.save(os.path.join(output_dir, image), img.format)
            except:
                print(image)
        if (i+1) % 100 == 0:
            print ("[{}/{}] Resized the images and saved into '{}'."
                   .format(i+1, num_images, output_dir))

def resize_image_operator(image_file, output_file, size, i, num_images):
    with open(image_file, 'r+b') as f:
        try:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(output_file, img.format)
        except Exception as e:
            print('problem', image_file, e)
    if (i + 1) % 10000 == 0:
        print("[{}/{}] Resized the images and saved."
              .format(i + 1, num_images))
    return

def resize_images_parallel(image_dir, size, pattern):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
            
    #num_cores = multiprocessing.cpu_count()
    num_cores = 6
    print('resize on {} CPUs'.format(num_cores))
    # Create imgs
    images = glob(os.path.join(image_dir, pattern))
    # Create dirs
    for image in images:
        output_file = image.replace('/images/','/resized_images/')
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
            
    num_images = len(images)
    Parallel(n_jobs=num_cores)(
        delayed(resize_image_operator)(
        image,
        image.replace('/images/','/resized_images/'),
        size,
        i,
        num_images) for i, image in enumerate(images))

if __name__ == '__main__':
    image_size = [256, 256]
    # Fashion IQ
    image_dir =  '../dataset/fashioniq/images/'
    pattern = '*.jpg'
    resize_images_parallel(image_dir, image_size, pattern)
    # Shoe
    image_dir =  '../dataset/shoe/images/'
    pattern = 'womens*/*/*.jpg'
    resize_images_parallel(image_dir, image_size, pattern)
    
