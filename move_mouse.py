import pyautogui
import time
import signal
import sys
import keyboard
# pip install pyautogui keyboard
# Hàm để xử lý khi nhận tín hiệu ngắt (Ctrl+C)
def signal_handler(sig, frame):
    print("Chương trình bị ngắt bởi người dùng.")
    sys.exit(0)

# Bắt tín hiệu ngắt (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

print("Nhấn Ctrl+C để dừng chương trình.")

# Vòng lặp vô hạn
try:
    while True:
        # Di chuyển chuột sang phải 100 pixel
        pyautogui.move(100, 0)
        time.sleep(1)
        
        # Di chuyển chuột sang trái 100 pixel
        pyautogui.move(-100, 0)
        time.sleep(1)
except KeyboardInterrupt:
    print("Chương trình bị ngắt bởi người dùng.")
    sys.exit(0)
