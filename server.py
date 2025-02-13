# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install transformers diffusers bitsandbytes flask flask-cors accelerate sentencepiece
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
import io
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusion3Pipeline

app = Flask(__name__, template_folder=".")
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
pipeline = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

@app.route("/")
def home():
    return render_template("new_index.html")

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        image_data = data.get("image")

        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
        
        image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        raw_image = raw_image.resize((256, 256))
        
        inputs = processor(raw_image, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        new_prompt = f"A highly detailed photorealistic image that is {caption}"

        generated_image = pipeline(
            prompt=new_prompt,
            num_inference_steps=4,  
            guidance_scale=0.0, 
        ).images[0]
        
        output_buffer = io.BytesIO()
        generated_image.save(output_buffer, format="PNG")
        encoded_image = base64.b64encode(output_buffer.getvalue()).decode("utf-8")

        return jsonify({
            "generated_image": encoded_image,
            "caption": caption
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)