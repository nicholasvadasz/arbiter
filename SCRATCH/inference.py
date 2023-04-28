import numpy as np
import chess
import cv2
import torch
import functools
import sys
import logging
import tensorflow as tf
from keras.optimizers import Adam

from pathlib import Path
from recap import URI, CfgNode as CN
from collections.abc import Iterable
from model_data.get_model import get_yolo, get_occupancy, get_corner
from utils import create_dataset as create_occupancy_dataset

tf.get_logger().setLevel(logging.ERROR)

### 1) MODEL INITIALIZATION (initialize all models with correct pathing, including YOLO)
# Devices
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

yolo_model, (occupancy_model, occupancy_cfg), corner_model = get_yolo(), get_occupancy(), get_corner()

if True:
    yolo_model.compile(optimizer=Adam(lr=1e-3), loss={
        # use custom yolo_loss Lambda layer.
        'yolo_loss': lambda y_true, y_pred: y_pred})

'''
### 2) INFERENCE CLASS (create class for inference, using chess API) // create modifiable state since we want to extend to video
class Arbiter:
    def __init__(self):
        self.squares = list(chess.SQUARES)
        self.log = []
        pass
    
    ### 3) OCCUPANCY CLASSIFICATION (using chesscog's functionality)
    def classify_occupancy():
    ## Chessboard initialization
        warped = create_occupancy_dataset.warp_chessboard_image(
                img, corners)
        square_imgs = map(functools.partial(
            create_occupancy_dataset.crop_square, warped, turn=turn), self._squares)
        square_imgs = map(Image.fromarray, square_imgs)
        square_imgs = map(self._occupancy_transforms, square_imgs)
        square_imgs = list(square_imgs)
        square_imgs = torch.stack(square_imgs)
        square_imgs = device(square_imgs)
        occupancy = self._occupancy_model(square_imgs)
        occupancy = occupancy.argmax(
            axis=-1) == self._occupancy_cfg.DATASET.CLASSES.index("occupied")
        occupancy = occupancy.cpu().numpy()
        return occupancy 


    ### 4) FULL CLASSIFICATION (combining occupancy and YOLO) // might need some helpers to abstract this into the class



def main(classifiers_folder: Path = URI("models://"), setup: callable = lambda: None):
    #Main method for running inference from the command line.

    #Args:
        #classifiers_folder (Path, optional): the path to the classifiers (supplying a different path is especially useful because the transfer learning classifiers are located at ``models://transfer_learning``). Defaults to ``models://``.
       # setup (callable, optional): An optional setup function to be called after the CLI argument parser has been setup. Defaults to lambda:None.
    

    parser = argparse.ArgumentParser(
        description="Run the chess recognition pipeline on an input image")
    parser.add_argument("file", help="path to the input image", type=str)
    parser.add_argument(
        "--white", help="indicate that the image is from the white player's perspective (default)", action="store_true", dest="color")
    parser.add_argument(
        "--black", help="indicate that the image is from the black player's perspective", action="store_false", dest="color")
    parser.set_defaults(color=True)
    args = parser.parse_args()

    setup()

    img = cv2.imread(str(URI(args.file)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    recognizer = ChessRecognizer(classifiers_folder)
    board, *_ = recognizer.predict(img, args.color)
    print()
    print(
        f"You can view this position at https://lichess.org/editor/{board.board_fen()}")

    if board.status() != Status.VALID:
        print()
        print("WARNING: The predicted chess position is not legal according to the rules of chess.")
        print("         You might want to try again with another picture.") '''