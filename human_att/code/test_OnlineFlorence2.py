from PIL import Image
import requests
import io, base64

task_prompt = "<OD>"
img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(img_url, stream=True).raw)

buf = io.BytesIO()
image.save(buf, format="JPEG")
img_64 = base64.b64encode(buf.getvalue()).decode("utf-8")

payload = {
    "task_prompt": task_prompt,
    "image_b64": img_64
}

res = requests.post(url="https://d8b67dcc50ee.ngrok-free.app/generate", json=payload)
print(res.json())