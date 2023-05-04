import numpy as np
import chess
import cv2
import torch
import functools
import logging
import tensorflow as tf
import os

from recap import URI, CfgNode as CN
from collections.abc import Iterable
from model_data.get_models import get_yolo, get_occupancy, get_corner
from model_data.yolo import YOLO as yolo
from utils import create_dataset as create_occupancy_dataset
from utils.transforms import build_transforms
from utils.datasets import Datasets
from utils.detect_corners import find_corners, resize_image
from utils.setup import device, detect_img, piece_dict
from PIL import Image

### Suppressing error messaging
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

### 1) MODEL INITIALIZATION (initialize all models with correct pathing, including YOLO)
yolo_model, (occupancy_model, occupancy_cfg), corner_cfg = yolo(), get_occupancy(), get_corner()

### 2) INFERENCE CLASS (create class for inference, using chess API) // create modifiable state since we want to extend to video
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

        cached_imgs = map(functools.partial(
            create_occupancy_dataset.crop_square, warped, turn=turn), self.squares)
        cached_imgs = map(Image.fromarray, cached_imgs)

        square_imgs = map(self.occupancy_transforms, square_imgs)
        square_imgs = list(square_imgs)

        square_imgs = torch.stack(square_imgs)
        square_imgs = device(square_imgs)
        occupancy = occupancy_model(square_imgs)
        occupancy = occupancy.argmax(
            axis=-1) == occupancy_cfg.DATASET.CLASSES.index("occupied")
        occupancy = occupancy.cpu().numpy()
        return occupancy, warped, cached_imgs
    
    def predict(self, img: np.ndarray, turn: chess.Color = chess.WHITE):
        with torch.no_grad():
            from timeit import default_timer as timer
            img, img_scale = resize_image(corner_cfg, img)
            t1 = timer()
            corners = find_corners(corner_cfg, img)
            occupancy, warped, cached_imgs = self.classify_occupancy(img, turn, corners)
            segmented_img, info = detect_img(yolo_model, Image.fromarray(warped))

            boxes, scores, box_scores, class_names, classes = info['boxes'], info['scores'], info['box_scores'], info['class_names'], info['classes']
            new_corners = find_corners(corner_cfg, warped)
            classify_dict = {}

            temp_occ = np.array(occupancy).reshape(8, 8)
            temp_occ = np.transpose(np.flip(np.flip(temp_occ, axis=0), axis=1))
            temp_occ = list(temp_occ.flatten())

            board = chess.Board()
            board.clear_board()
            for box, label, score in zip(boxes, classes, scores):
                located_square = create_occupancy_dataset.find_square(box, new_corners)
                if located_square is None:
                    pass
                else:
                    piece = piece_dict(class_names[label])
                    file, rank = (int) (located_square/8), (located_square % 8) - 1
                    if located_square not in classify_dict:
                        classify_dict[located_square] = score
                        board.set_piece_at(chess.square(rank, file), piece)
                    else:
                        if score > classify_dict[located_square]:
                            classify_dict[located_square] = score
                            board.set_piece_at(chess.square(rank, file), piece)            

            yolo_model.close_session()
            t2 = timer()
            print(t2 - t1)

            return board, warped, segmented_img, temp_occ
        
    def piecewise_predict(self, img: np.ndarray, turn: chess.Color = chess.WHITE):
        '''
        Poor performance when we predict square by square, showing that our model is likely
        scale-sensitive -> could improve by augmenting training dataset.
        '''
        with torch.no_grad():
            from timeit import default_timer as timer
            t1 = timer()
            corners = find_corners(corner_cfg, img)
            occupancy, warped, cached_imgs = self.classify_occupancy(img, turn, corners)

            for idx, (occupied, square) in enumerate(zip(occupancy, cached_imgs)):
                print(f'Value: {idx}  Occupancy: {occupied}')
                square = square.resize((1200, 1200))
                segmented_img, info = detect_img(yolo_model, square)
                if occupied:
                    segmented_img.show()
            yolo_model.close_session()
        
    def video_predict(video_path, output_path):
        """ import timer
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")
        
        video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
        video_fps       = vid.get(cv2.CAP_PROP_FPS)
        video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)

        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        while True:
            return_value, frame = vid.read()
            image = Image.fromarray(frame)
            image, _ = yolo.detect_image(image)
            result = np.asarray(image)
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
            if isOutput:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        yolo.close_session() """
        pass

        
if __name__ == '__main__':   
    img = cv2.imread('testing.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    recognizer = Arbiter()
    board, warped, segmented_img, temp_occ = recognizer.predict(img)
    print(f"You can view this position at https://lichess.org/editor/{board.board_fen()}")
    segmented_img.show()




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