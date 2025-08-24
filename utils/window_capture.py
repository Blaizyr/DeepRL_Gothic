# utils/window_capture.py
import win32gui
import win32ui
import numpy as np
from ctypes import windll

def find_window_by_title_substring(substr: str):
    hwnds = []
    def _enum(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if substr.lower() in title.lower():
                hwnds.append(hwnd)
    win32gui.EnumWindows(_enum, None)
    return hwnds[0] if hwnds else None

def grab_window(hwnd, bbox=None):
    # bbox=(left, top, right, bottom) in screen coords; if None → full client rect
    if hwnd is None:
        return None
    # client rect
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    # Map to screen
    left_top = win32gui.ClientToScreen(hwnd, (left, top))
    right_bottom = win32gui.ClientToScreen(hwnd, (right, bottom))
    L, T = left_top
    R, B = right_bottom
    if bbox is not None:
        L, T, R, B = bbox

    w = R - L
    h = B - T
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
    saveDC.SelectObject(saveBitMap)
    result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 1)

    bmpinfo = saveBitMap.GetInfo()
    bmpstr  = saveBitMap.GetBitmapBits(True)
    img = np.frombuffer(bmpstr, dtype=np.uint8)
    if h == 0 or bmpinfo['bmWidthBytes'] == 0:
        print("[WARN] Grab_window zwrócił pustą ramkę")
        return None
    img.shape = (h, bmpinfo['bmWidthBytes']//4, 4)
    img = img[:, :w, :3]  # BGR
    img = img[..., ::-1]  # to RGB

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    if result != 1:
        return None
    return img
