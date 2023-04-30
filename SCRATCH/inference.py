import numpy as np
import chess
import cv2
import torch
import functools
import sys
import logging
import tensorflow as tf
import typing

from keras.optimizers import Adam
from pathlib import Path
from recap import URI, CfgNode as CN
from collections.abc import Iterable
from chess import Status
from model_data.get_model import get_yolo, get_occupancy, get_corner
from model_data.yolo import YOLO as yolo
from utils import create_dataset as create_occupancy_dataset
from utils.transforms import build_transforms
from utils.datasets import Datasets
from utils.detect_corners import find_corners, resize_image
from PIL import Image

tf.get_logger().setLevel(logging.ERROR)

### 1) MODEL INITIALIZATION (initialize all models with correct pathing, including YOLO)
# Devices
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

T = typing.Union[torch.Tensor, torch.nn.Module, typing.List[torch.Tensor],
                 tuple, dict, typing.Generator]

def device(x: T, dev: str = DEVICE) -> T:
    """Convenience method to move a tensor/module/other structure containing tensors to the device.

    Args:
        x (T): the tensor (or strucure containing tensors)
        dev (str, optional): the device to move the tensor to. Defaults to DEVICE.

    Raises:
        TypeError: if the type was not a compatible tensor

    Returns:
        T: the input tensor moved to the device
    """

    to = functools.partial(device, dev=dev)
    if isinstance(x, (torch.Tensor, torch.nn.Module)):
        return x.to(dev)
    elif isinstance(x, list):
        return list(map(to, x))
    elif isinstance(x, tuple):
        return tuple(map(to, x))
    elif isinstance(x, dict):
        return {k: to(v) for k, v in x.items()}
    elif isinstance(x, Iterable):
        return map(to, x)
    else:
        raise TypeError

yolo_model, (occupancy_model, occupancy_cfg), corner_cfg = yolo(), get_occupancy(), get_corner()

""" if True:
    yolo_model.compile(optimizer=Adam(lr=1e-3), loss={
        # use custom yolo_loss Lambda layer.
        'yolo_loss': lambda y_true, y_pred: y_pred}) """


### 2) INFERENCE CLASS (create class for inference, using chess API) // create modifiable state since we want to extend to video
def detect_img(yolo, path):
    end = True
    while end:
        img = path
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
            end = False
    yolo.close_session()

class Arbiter:
    def __init__(self):
        self.squares = list(chess.SQUARES)
        self.log = []
        self.occupancy_transforms = build_transforms(
            occupancy_cfg, mode=Datasets.TEST)

    ### 3) OCCUPANCY CLASSIFICATION (using chesscog's functionality)
    def classify_occupancy(self, img: np.ndarray, turn: chess.Color, corners: np.ndarray) -> np.ndarray:
    ## Chessboard initialization
        warped = create_occupancy_dataset.warp_chessboard_image(
                img, corners)
        square_imgs = map(functools.partial(
            create_occupancy_dataset.crop_square, warped, turn=turn), self.squares)
        square_imgs = map(Image.fromarray, square_imgs)
        square_imgs = map(self.occupancy_transforms, square_imgs)
        square_imgs = list(square_imgs)
        square_imgs = torch.stack(square_imgs)
        square_imgs = device(square_imgs)
        occupancy = occupancy_model(square_imgs)
        occupancy = occupancy.argmax(
            axis=-1) == occupancy_cfg.DATASET.CLASSES.index("occupied")
        occupancy = occupancy.cpu().numpy()
        return occupancy, warped
    
    def predict(self, img: np.ndarray, turn: chess.Color = chess.WHITE):
        with torch.no_grad():
            from timeit import default_timer as timer
            img, img_scale = resize_image(corner_cfg, img)
            t1 = timer()
            corners = find_corners(corner_cfg, img)
            occupancy, warped = self.classify_occupancy(img, turn, corners)
            Image.fromarray(warped).save("warped.jpg")
            img = Image.open("warped.jpg")
            segmented_image = detect_img(yolo_model, "warped.jpg")
            t2 = timer()
            print(t2 - t1)

            '''board = chess.Board()
            board.clear_board()
            for square, piece in zip(self._squares, pieces):
                if piece:
                    board.set_piece_at(square, piece)
            corners = corners / img_scale'''
            return occupancy, segmented_image
        
    def video_predict():
        pass

        
    
        
img = cv2.imread('test2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
recognizer = Arbiter()
occupancy = recognizer.predict(img)
print(occupancy)




    ### 4) FULL CLASSIFICATION (combining occupancy and YOLO) // might need some helpers to abstract this into the class

'''

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