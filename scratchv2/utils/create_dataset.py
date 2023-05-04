"""Script to create the occupancy classification dataset.

Based on the four chessboard corner points which are supplied as labels, this module is responsible for warping the image and cutting out the squares.
Note that before running this module requires the rendered dataset to be downloaded and split (see the :mod:`chesscog.data_synthesis` module for more information).

.. code-block:: console

    $ python -m chesscog.occupancy_classifier.create_dataset --help
    usage: create_dataset.py [-h]
    
    Create the dataset for occupancy classification.
    
    optional arguments:
      -h, --help  show this help message and exit
"""

from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import json
import numpy as np
import chess
import os
import shutil
from recap import URI
import argparse

RENDERS_DIR = None
OUT_DIR = None
SQUARE_SIZE = 50
BOARD_SIZE = 8 * SQUARE_SIZE
IMG_SIZE = BOARD_SIZE + 2 * SQUARE_SIZE

def sort_corner_points(points: np.ndarray) -> np.ndarray:
    """Permute the board corner coordinates to the order [top left, top right, bottom right, bottom left].

    Args:
        points (np.ndarray): the four corner coordinates

    Returns:
        np.ndarray: the permuted array
    """

    # First, order by y-coordinate
    points = points[points[:, 1].argsort()]
    # Sort top x-coordinates
    points[:2] = points[:2][points[:2, 0].argsort()]
    # Sort bottom x-coordinates (reversed)
    points[2:] = points[2:][points[2:, 0].argsort()[::-1]]

    return points


def crop_square(img: np.ndarray, square: chess.Square, turn: chess.Color) -> np.ndarray:
    """Crop a chess square from the warped input image for occupancy classification.

    Args:
        img (np.ndarray): the warped input image
        square (chess.Square): the square to crop
        turn (chess.Color): the current player

    Returns:
        np.ndarray: the cropped square
    """
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    if turn == chess.WHITE:
        row, col = 7 - rank, file
    else:
        row, col = rank, 7 - file
    return img[int(SQUARE_SIZE * (row + .5)): int(SQUARE_SIZE * (row + 2.5)),
               int(SQUARE_SIZE * (col + .5)): int(SQUARE_SIZE * (col + 2.5))]

def find_square(box, new_corners):
    y0, x0, y1, x1 = box
    center_x, center_y = (x0 + x1) / 2, (y0 + (5*y1)) / 6
    avg_left = (new_corners[0][0] + new_corners[3][0]) / 2
    avg_right = (new_corners[1][0] + new_corners[2][0]) / 2
    avg_top = (new_corners[0][1] + new_corners[1][1]) / 2
    avg_bottom = (new_corners[2][1] + new_corners[3][1]) / 2
    rank = 7 - int(8*(center_x - avg_left) / (avg_right - avg_left))
    file = int(8*(center_y - avg_top) / (avg_bottom - avg_top))
    
    if rank not in range(0, 8) or file not in range(0, 8):
        return None
    else:
        return chess.square(file, rank)


def warp_chessboard_image(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp the image of the chessboard onto a regular grid.

    Args:
        img (np.ndarray): the image of the chessboard
        corners (np.ndarray): pixel locations of the four corner points

    Returns:
        np.ndarray: the warped image
    """

    src_points = sort_corner_points(corners)
    dst_points = np.array([[SQUARE_SIZE, SQUARE_SIZE],  # top left
                           [BOARD_SIZE + SQUARE_SIZE, SQUARE_SIZE],  # top right
                           [BOARD_SIZE + SQUARE_SIZE, BOARD_SIZE + \
                            SQUARE_SIZE],  # bottom right
                           [SQUARE_SIZE, BOARD_SIZE + SQUARE_SIZE]  # bottom left
                           ], dtype=np.float32)
    transformation_matrix, mask = cv2.findHomography(src_points, dst_points)
    return cv2.warpPerspective(img, transformation_matrix, (IMG_SIZE, IMG_SIZE))


def _extract_squares_from_sample(id: str, subset: str = "", input_dir: Path = RENDERS_DIR, output_dir: Path = OUT_DIR):
    img = cv2.imread(str(input_dir / subset / (id + ".png")))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with (input_dir / subset / (id + ".json")).open("r") as f:
        label = json.load(f)

    corners = np.array(label["corners"], dtype=np.float32)
    unwarped = warp_chessboard_image(img, corners)

    board = chess.Board(label["fen"])

    for square in chess.SQUARES:
        target_class = "empty" if board.piece_at(
            square) is None else "occupied"
        piece_img = crop_square(unwarped, square, label["white_turn"])
        with Image.fromarray(piece_img, "RGB") as piece_img:
            piece_img.save(output_dir / subset / target_class /
                           f"{id}_{chess.square_name(square)}.png")


def create_dataset(input_dir: Path = RENDERS_DIR, output_dir: Path = OUT_DIR):
    """Create the occupancy classification dataset.

    Args:
        input_dir (Path, optional): the input folder of the rendered images. Defaults to ``data://render``.
        output_dir (Path, optional): the output folder. Defaults to ``data://occupancy``.
    """
    for subset in ("train", "val", "test"):
        for c in ("empty", "occupied"):
            folder = output_dir / subset / c
            shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder, exist_ok=True)
        samples = list((input_dir / subset).glob("*.png"))
        for i, img_file in enumerate(samples):
            if len(samples) > 100 and i % int(len(samples) / 100) == 0:
                print(f"{i / len(samples)*100:.0f}%")
            _extract_squares_from_sample(img_file.stem, subset,
                                         input_dir, output_dir)


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Create the dataset for occupancy classification.").parse_args()
    create_dataset()
