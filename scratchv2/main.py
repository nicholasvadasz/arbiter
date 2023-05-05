import time
import cv2
import PySimpleGUI as sg
import inference
import chess
import sys
from io import StringIO

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

colwebcam1_layout = [[sg.Image(filename="", key="cam1")]]
colwebcam1 = sg.Column(colwebcam1_layout, element_justification='center')

boardcol = sg.Column(board, element_justification='center', key="board")
colslayout = [boardcol, colwebcam1]

layout = [colslayout]

window = sg.Window("Arbiter", layout, 
                    no_titlebar=False, alpha_channel=1, grab_anywhere=False, 
                    return_keyboard_events=True, location=(100, 100))        

while True:
    start_time = time.time()
    event, values = window.read(timeout=20)

    if event == sg.WIN_CLOSED:
        break

    ret, frameOrig = video_capture.read()
    frame = cv2.resize(frameOrig, frameSize)
    imgbytes = cv2.imencode(".png", frame)[1].tobytes()
    window["cam1"].update(data=imgbytes)
    time_since_last_update += time.time() - start_time
    if time_since_last_update > 5:
        # returnBoard = inference.videopipeline()
        # time.sleep(3)
        # with Capturing() as output:
        #     print(returnBoard)
        # newboard_str = ""
        # for i in range(8):
        #     for j in range(15):
        #         if output[i][j] == '.':
        #             newboard_str += '.'
        #         if output[i][j] == ' ':
        #             pass
        #         else:
        #             newboard_str += output[i][j]
        # for i in range(8):
        #     for j in range(8):
        #         if board_str[i*8+j] != newboard_str[i*8+j]:
        #             if newboard_str[i*8+j] == '.':
        #                 board[i][j].update(image_filename=f'./pieces/empty.png')
        #             else:
        #                 piece = newboard_str[i*8+j]
        #                 white = piece.isupper()
        #                 color = 'white' if white else 'black'
        #                 board[i][j].update(image_filename=f'./pieces/{color}{piece.lower()}.png')
        window["board"].update(board)
        window.refresh()        
        time_since_last_update = 0

video_capture.release()
cv2.destroyAllWindows()