import re
import ffmpeg
from tqdm import tqdm
import os
from pathlib import Path

CHUNK_SIZE = 3500


def convert_movies_to_images(dirname: str, data_key: str, full_run: bool = False, verbose: bool = False):
    """
    Converts MP4 videos from the dataset directory to JPG images.
    :param dirname: Path to the dataset root directory
    :param data_key: key, which identifies the data group (e.g. s4-d1)
    :param full_run: (bool, optional, default=False) whether the conversion should be run in full, without skipping already converted folders.
    :param verbose: (bool, optional, default=False) whether detailed logs should be printed.
    :return: None
    """
    movie_dir = dirname + f'{data_key}/movies/'

    movie_rgx = re.compile(f'{data_key}-camera(.)-(.*).mp4')

    movie_files = os.listdir(movie_dir)
    for movie_file in tqdm(movie_files, desc='Converting MP4 to JPG'):
        match = re.search(movie_rgx, movie_file)
        camera_id = match.group(1)
        chunk = match.group(2)

        output_image_dir = Path(dirname + f'{data_key}/images/{data_key}-camera{camera_id}-{chunk}')

        output_image_dir.mkdir(parents=True, exist_ok=True)

        # Check whether the conversion of movie was complete. If yes skip conversion.
        existing_images = os.listdir(output_image_dir)
        if len(existing_images) == CHUNK_SIZE and not full_run:
            continue

        full_movie_path = movie_dir + movie_file
        stream = ffmpeg.input(full_movie_path)
        stream = ffmpeg.output(stream,
                               dirname + f"{data_key}/images/{data_key}-camera{camera_id}-{chunk}/{data_key}-camera{camera_id}-%05d.jpg",
                               format='image2',
                               vcodec='mjpeg')
        stream.run(quiet=verbose)

    print(f'Images have been saved to: {dirname}/{data_key}/images/')
