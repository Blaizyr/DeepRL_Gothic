import numpy as np
import win32gui
import win32ui
from ctypes import windll
import cv2

class ScreenCapture:
    def __init__(self, window_title_substr, frame_shape=(84,84), crop=None, gray=True, normalize=True):
        self.window_title_substr = window_title_substr
        self.frame_shape = frame_shape
        self.crop = crop
        self.gray = gray
        self.normalize = normalize
        self.hwnd = None
        self.prev_obs = None

    def _find_window(self):
        hwnds = []
        def _enum(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if self.window_title_substr.lower() in title.lower():
                    hwnds.append(hwnd)
        win32gui.EnumWindows(_enum, None)
        self.hwnd = hwnds[0] if hwnds else None

    def _grab_window(self):
        if self.hwnd is None:
            self._find_window()
            if self.hwnd is None:
                return None

        left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
        left_top = win32gui.ClientToScreen(self.hwnd, (left, top))
        right_bottom = win32gui.ClientToScreen(self.hwnd, (right, bottom))
        L, T = left_top
        R, B = right_bottom

        w, h = R - L, B - T
        if w == 0 or h == 0:
            return None

        hwndDC = win32gui.GetWindowDC(self.hwnd)
        mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
        saveDC.SelectObject(saveBitMap)

        try:
            result = windll.user32.PrintWindow(self.hwnd, saveDC.GetSafeHdc(), 1)
            bmpinfo = saveBitMap.GetInfo()
            bmpstr  = saveBitMap.GetBitmapBits(True)
            img = np.frombuffer(bmpstr, dtype=np.uint8)
            img.shape = (h, bmpinfo['bmWidthBytes']//4, 4)
            img = img[:, :w, :3][..., ::-1]  # RGB
            return img if result == 1 else None
        finally:
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, hwndDC)

    def _preprocess(self, img):
        if img is None:
            return None
        if self.crop is not None:
            y0, y1, x0, x1 = self.crop
            img = img[y0:y1, x0:x1]
        img = cv2.resize(img, self.frame_shape, interpolation=cv2.INTER_AREA)
        if self.gray:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=-1)
        if self.normalize:
            img = img.astype(np.float32)/255.0
        else:
            img = img.astype(np.uint8)
        return img

    def get_observation(self, preprocess=True, force_update=False):
        img = self._grab_window()
        if not preprocess:
            return img if img is not None else self.prev_obs
        obs = self._preprocess(img)
        if obs is None:
            return self.prev_obs if self.prev_obs is not None else np.zeros((*self.frame_shape, 1), dtype=np.uint8)
        if force_update or obs is not None:
            self.prev_obs = obs
        return obs
