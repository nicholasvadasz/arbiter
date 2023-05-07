import time
import cv2
import PySimpleGUI as sg
import inference
import chess
import sys
from io import StringIO
from stockfish import Stockfish

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

# Camera Settings
camera_Width  = 320 # 480 # 640 # 1024 # 1280
camera_Heigth = 240 # 320 # 480 # 780  # 960
frameSize = (camera_Width, camera_Heigth)
video_capture = cv2.VideoCapture(0)
time.sleep(2.0)

stock = Stockfish("../../Stockfish/src/stockfish")
stock.set_fen_position("rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
total_eval = stock.get_evaluation()
centipawn = float(total_eval["value"]) / 100  
negative_or_positive = "+" if centipawn > 0 else ""

sg.theme("Dark")

time_since_last_update = 0
start_time = time.time()

def make_board(board_str):
    board = [[sg.Button(size=(1,1), pad=(0,0), button_color=('black','white'), key=(i,j), image_filename=f'./pieces/empty.png') if (i+j)%2 == 0 else sg.Button(size=(1,1), pad=(0,0), button_color=('white','black'), key=(i,j), image_filename=f'./pieces/empty.png') for j in range(8)] for i in range(8)]
    for i in range(8):
        for j in range(8):
            if board_str[i*8+j] != '.':
                piece = board_str[i*8+j]
                white = piece.isupper()
                color = 'white' if white else 'black'
                board[i][j] = sg.Button(size=(1,1), pad=(0,0), button_color=('black','white'), key=(i,j), image_filename=f'./pieces/{color}{piece.lower()}.png') if (i+j)%2 == 0 else sg.Button(size=(1,1), pad=(0,0), button_color=('white','black'), key=(i,j), image_filename=f'./pieces/{color}{piece.lower()}.png')
    return board

board_str = "rnbqkbnrpppppppp................................PPPPPPPPRNBQKBNR"
board = make_board(board_str)

evalFont = ("Noto Sans Lydian", 20)
evaluation_layout = [[sg.Text("EVALUATION:", font=evalFont), sg.Text(f"{negative_or_positive}{centipawn:.2f}", font=evalFont, key="eval")]]

# put evaluation under webcam
colwebcam1_layout = [[sg.Image(filename="", key="cam1")],
                    [sg.Column(evaluation_layout, element_justification='center', key="evaluation")]]
colwebcam1 = sg.Column(colwebcam1_layout, element_justification='center', key="webcam1")

boardcol = sg.Column(board, element_justification='center', key="board")
colslayout = [boardcol, colwebcam1]

layout = [colslayout]

window = sg.Window("Arbiter", layout, 
                    no_titlebar=False, alpha_channel=1, grab_anywhere=False, 
                    return_keyboard_events=True, location=(100, 100))        

while True:
    event, values = window.read(timeout=20)

    if event == sg.WIN_CLOSED:
        break

    ret, frameOrig = video_capture.read()
    frame = cv2.resize(frameOrig, frameSize)
    imgbytes = cv2.imencode(".png", frame)[1].tobytes()
    window["cam1"].update(data=imgbytes)
    if event == 'p':
        returnBoard = inference.videopipeline()
        stock.set_fen_position(returnBoard.board_fen())
        total_eval = stock.get_evaluation()
        centipawn = float(total_eval["value"]) / 100
        negative_or_positive = "+" if centipawn > 0 else ""
        window["eval"].update(f"{negative_or_positive}{centipawn:.2f}")

        with Capturing() as output:
            print(returnBoard)
        newboard_str = ""
        for i in range(8):
            tempoutput = output[i]
            tempoutput = tempoutput.replace(" ", "")
            tempoutput = tempoutput[::-1]
            for j in range(0, len(tempoutput)):
                newboard_str += tempoutput[j]
        for i in range(8):
            for j in range(8):
                board[i][j].update(image_filename=f'./pieces/empty.png')
        for i in range(8):
            for j in range(8):
                if newboard_str[i*8+j] == '.':
                    board[i][j].update(image_filename=f'./pieces/empty.png')
                else:
                    piece = newboard_str[i*8+j]
                    white = piece.isupper()
                    color = 'white' if white else 'black'
                    board[i][j].update(image_filename=f'./pieces/{color}{piece.lower()}.png')
        window["board"].update(board)
        window.refresh()        
        time_since_last_update = 0
        output = []
        returnBoard = []

video_capture.release()
cv2.destroyAllWindows()