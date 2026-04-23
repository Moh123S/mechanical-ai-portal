from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch

def analyze_mechanical_part(image_path):
    # Model ID (Chota aur tez model)
    model_id = "vikhyatk/moondream2"
    revision = "2024-08-26"
    
    print("⏳ Model load ho raha hai (Pehli baar thoda time lega)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    
    image = Image.open(image_path)
    # Image ko scan karo
    enc_image = model.encode_image(image)
    
    # Prompt
    question = "Identify this industrial equipment and describe its technical condition."
    answer = model.answer_question(enc_image, question, tokenizer)
    
    return answer

if __name__ == "__main__":
    try:
        print("🚀 Local Vision Mode: No API Key needed!")
        analysis = analyze_mechanical_part("test_part.jpg")
        print("\n🛠️ --- ANALYSIS REPORT ---")
        print(analysis)
    except Exception as e:
        print(f"❌ Glitch: {e}")