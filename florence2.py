import requests
import time
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
print("finish loading model")
prompt = "<DETAILED_CAPTION>"

url = "/home/coder/human-attribute-recognition/static/camera_2_20241101T071432474160_0.jpg"
image = Image.open(url)

inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    num_beams=3,
    do_sample=False
)
# Define the number of iterations
num_iterations = 10

# Initialize variables to store performance data
total_time = 0
iteration_times = []
print("start detecting")
start_process = time.time() 
# Loop to benchmark the process
for i in range(num_iterations):
    start_time = time.time()
    
    # Your processing code
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task="<DETAILED_CAPTION>", image_size=(image.width, image.height))
    
    # Calculate elapsed time for this iteration
    elapsed_time = time.time() - start_time
    iteration_times.append(elapsed_time)
    total_time += elapsed_time

    # Print details for this iteration
    print(f"Iteration {i + 1}: Time taken = {elapsed_time:} seconds")
    print(parsed_answer)
whole_time = time.time() - start_process
print(f"whole time: {whole_time}")
# Calculate and display overall performance metrics
average_time = total_time / num_iterations
print(f"\nTotal time for {num_iterations} iterations: {total_time:} seconds")
print(f"Average time per iteration: {average_time:} seconds")
print(f"All iteration times: {iteration_times}")
