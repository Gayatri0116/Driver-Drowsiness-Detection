# Driver-Drowsiness-Detection
Real-time driver drowsiness detection using facial landmarks, EAR, MAR, and alerts


# Driver Drowsiness Detection using Facial Landmarks

This project detects driver drowsiness using:
- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- Yawning detection
- Visual + Audio Alerts

## 🧠 Technologies Used
- Python
- OpenCV
- Dlib
- imutils
- NumPy

## ⚙️ How to Run

1. Clone the repo:
    ```bash
    git clone https://github.com/yourusername/Driver-Drowsiness-Detection.git
    cd Driver-Drowsiness-Detection
    ```

2. Create virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    venv\Scripts\activate  # on Windows
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download Dlib's shape predictor model:  
   [Download Here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)  
   Extract and place it inside `dlib_shape_predictor/`.

5. Run the script:
    ```bash
    python Driver_Drowsiness_Detection.py
    ```

## 📝 Acknowledgements
- Dlib for the facial landmark model
- Adrian Rosebrock for early concepts on EAR/MAR (PyImageSearch)


