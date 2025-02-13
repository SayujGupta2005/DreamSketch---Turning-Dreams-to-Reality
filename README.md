# ⭐ DreanSketch - Turning Dreams to Reality ⭐
# Author: Sayuj Gupta
# Date: 13-2-2025

## Overview
This project allows users to draw an image on a web-based whiteboard and send it to a backend, where a deep learning model processes the drawing to generate a high-quality image using Stable Diffusion XL (SDXL) and BLIP captioning.

## Reason of Creation
This project was created for an event in my college-IIT Jammu (Hence the logo)

## Requirements
### 1. Hugging Face CLI Login
To access Hugging Face models, authenticate using the CLI:
```bash
huggingface-cli login
```
You will need an access token from [Hugging Face](https://huggingface.co/settings/tokens).

### 2. Hardware Requirements
A CUDA-enabled GPU with at least **8GB VRAM** is required to run Stable Diffusion efficiently. NVIDIA RTX 3060 or better is recommended.

### 3. Prerequisites
Ensure you have **Python 3.8.10** installed. If not, install it from [Python's official site](https://www.python.org/downloads/).

Install the required dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers diffusers bitsandbytes flask flask-cors accelerate sentencepiece
pip install opencv-python numpy pillow
pip install huggingface_hub
```

## Usage

### 1. Running the Backend Server
Start the Flask-based backend that process images:
```bash
python server.py
```
This will launch the server, allowing the frontend to communicate with the deep learning model.

### 2. Running the Frontend
The frontend consists of a whiteboard for sketching and sending images to the backend. Deploy it using a simple server:
```bash
python -m http.server 5000
```
Now, visit `http://localhost:8000` in your browser.

## How It Works
1. **User Sketches an Image** – The frontend provides a whiteboard where users can draw a rough sketch and press generate.
2. **Image Sent to Backend** – The sketch is sent to the Flask backend via an API request.
3. **BLIP Captioning** – The sketch undergoes captioning using the BLIP model to generate a meaningful description.
4. **Stable Diffusion XL (SDXL) Processing** – The captioned text is fed into SDXL, which generates a high-quality image.
5. **Image Display** – The generated image is sent back to the frontend and displayed to the user.

## API Endpoints
### 1. Upload Sketch & Generate Image
- **URL:** `/generate`
- **Method:** `POST`
- **Payload:** Sketch image (base64 encoded or multipart form-data)
- **Response:** AI-generated image

### 2. Health Check
- **URL:** `/health`
- **Method:** `GET`
- **Response:** `{ "status": "running" }`

## Notes
- When 'server.py' is run for first time, it takes around 5 minutes to start everything
- Image generation takes around 2-3 minutes depending on your gpu.

## Future Improvements
- Add support for multiple sketch styles.
- Improve captioning accuracy with fine-tuned BLIP models.
- Optimize model inference for better performance on lower-end GPUs.

## License
This project is released under the MIT License.

## Contributing
If you’d like to contribute, fork the repository and submit a pull request. Suggestions and improvements are welcome!

## Contact
For any issues or questions, feel free to reach out at [sayujgupta2005@gmail.com](mailto:sayujgupta2005@.com) or create an issue in the repository.
