# install_windows.ps1
param(
  [string]$Python="python"
)

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
  . .\.venv\Scripts\Activate.ps1
} else {
  & $Python -m venv .venv
  . .\.venv\Scripts\Activate.ps1
}

python -m pip install --upgrade pip

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install gymnasium==0.29.1 stable-baselines3[extra]==2.3.2
pip install numpy opencv-python mss pywin32 pydirectinput keyboard matplotlib tqdm scikit-learn
