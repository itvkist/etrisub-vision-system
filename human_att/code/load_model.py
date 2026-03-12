import torch
from transformers import AutoProcessor, AutoModelForCausalLM 


# Device configuration
# ------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Set data type
torch_dtype = torch.float32

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large",
    torch_dtype=torch_dtype,  # Ensure to match the saved dtype
    trust_remote_code=True
).to(DEVICE)

# Load the processor
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-large",
    trust_remote_code=True
)