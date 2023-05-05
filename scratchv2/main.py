import time
import cv2
import PySimpleGUI as sg

# Camera Settings
camera_Width  = 320 # 480 # 640 # 1024 # 1280
camera_Heigth = 240 # 320 # 480 # 780  # 960
frameSize = (camera_Width, camera_Heigth)
video_capture = cv2.VideoCapture(0)
time.sleep(2.0)

# init Windows Manager
sg.theme("DarkBlue")

def make_board():
    board = [[sg.Button(size=(2,2), pad=(0,0), button_color=('black','white'), key=(i,j)) if (i+j)%2 == 0 else sg.Button(size=(2,2), pad=(0,0), button_color=('white','black'), key=(i,j)) for j in range(8)] for i in range(8)]
    return board

board = make_board()

colwebcam1_layout = [[sg.Image(filename="", key="cam1")]]
colwebcam1 = sg.Column(colwebcam1_layout, element_justification='center')

boardcol = sg.Column(board, element_justification='center')
colslayout = [boardcol, colwebcam1]

layout = [colslayout]

window    = sg.Window("Arbiter", layout, 
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
    
  

video_capture.release()
cv2.destroyAllWindows()