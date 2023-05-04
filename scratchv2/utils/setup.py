import torch
import typing
import functools
import chess

from collections.abc import Iterable
from PIL import Image

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

### 2) INFERENCE CLASS (create class for inference, using chess API) // create modifiable state since we want to extend to video
def detect_img(yolo, path):
    end = True
    while end:
        try:
            image = Image.open(path)
        except:
            image = path
        finally:
            r_image, info = yolo.detect_image(image)
            end = False
    return r_image, info


def piece_dict(str):
        if str == 'bishop':
            return chess.Piece(chess.BISHOP, chess.WHITE)
        if str == 'black-bishop':
            return chess.Piece(chess.BISHOP, chess.BLACK)
        if str == 'black-king':
            return chess.Piece(chess.KING, chess.BLACK)
        if str == 'black-knight':
            return chess.Piece(chess.KNIGHT, chess.BLACK)
        if str == 'black-pawn':
            return chess.Piece(chess.PAWN, chess.BLACK)
        if str == 'black-queen':
            return chess.Piece(chess.QUEEN, chess.BLACK)
        if str == 'black-rook':
            return chess.Piece(chess.ROOK, chess.BLACK)
        if str == 'white-bishop':
            return chess.Piece(chess.BISHOP, chess.WHITE)
        if str == 'white-king':
            return chess.Piece(chess.KING, chess.WHITE)
        if str == 'white-knight':
            return chess.Piece(chess.KNIGHT, chess.WHITE)
        if str == 'white-pawn':
            return chess.Piece(chess.PAWN, chess.WHITE)
        if str == 'white-queen':
            return chess.Piece(chess.QUEEN, chess.WHITE)
        if str == 'white-rook':
            return chess.Piece(chess.ROOK, chess.WHITE)
        else:
            return None