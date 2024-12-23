import gradio as gr
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, pipeline
from sentence_transformers import SentenceTransformer, util
import requests
import random
import warnings
from transformers import logging
import os
import tensorflow as tf
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(".env")

# Set environment configurations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set. Please add it to the Secrets in your Hugging Face Space settings.")

def segment_into_sentences_groq(passage):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {
                "role": "system",
                "content": "you are to segment the sentence by adding '1!2@3#' at the end of each sentence. Return only the segmented sentences only return the modified passage and nothing else do not add your responses"
            },
            {
                "role": "user",
                "content": f"you are to segment the sentence by adding '1!2@3#' at the end of each sentence. Return only the segmented sentences only return the modified passage and nothing else do not add your responses. here is the passage:{passage}"
            }
        ],
        "temperature": 1.0,
        "max_tokens": 8192
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        try:
            segmented_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            sentences = segmented_text.split("1!2@3#")
            return [sentence.strip() for sentence in sentences if sentence.strip()]
        except (IndexError, KeyError):
            raise ValueError("Unexpected response structure from Groq API.")
    else:
        raise ValueError(f"Groq API error: {response.text}")

class TextEnhancer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.paraphrase_tokenizer = AutoTokenizer.from_pretrained("prithivida/parrot_paraphraser_on_T5")
        self.paraphrase_model = T5ForConditionalGeneration.from_pretrained("prithivida/parrot_paraphraser_on_T5").to(self.device)
        self.grammar_pipeline = pipeline(
            "text2text-generation",
            model="Grammarly/coedit-large",
            device=0 if self.device == "cuda" else -1
        )
        self.similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(self.device)

    def enhance_text(self, text, min_similarity=0.8, max_variations=3):
        sentences = segment_into_sentences_groq(text)
        enhanced_sentences = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            inputs = self.paraphrase_tokenizer(
                f"paraphrase: {sentence}",
                return_tensors="pt",
                padding=True,
                max_length=150,
                truncation=True
            ).to(self.device)
            outputs = self.paraphrase_model.generate(
                **inputs,
                max_length=len(sentence.split()) + 20,
                num_return_sequences=max_variations,
                num_beams=max_variations,
                temperature=0.7
            )
            paraphrases = [
                self.paraphrase_tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            sentence_embedding = self.similarity_model.encode(sentence)
            paraphrase_embeddings = self.similarity_model.encode(paraphrases)
            similarities = util.cos_sim(sentence_embedding, paraphrase_embeddings)
            valid_paraphrases = [
                para for para, sim in zip(paraphrases, similarities[0])
                if sim >= min_similarity
            ]
            if valid_paraphrases:
                corrected = self.grammar_pipeline(
                    valid_paraphrases[0],
                    max_length=150,
                    num_return_sequences=1
                )[0]["generated_text"]
                enhanced_sentences.append(corrected)
            else:
                enhanced_sentences.append(sentence)
        
        enhanced_text = ". ".join(sentence.rstrip(".") for sentence in enhanced_sentences) + "."
        return enhanced_text

def create_interface():
    enhancer = TextEnhancer()
    
    def process_text(text, similarity_threshold):
        try:
            return enhancer.enhance_text(
                text,
                min_similarity=similarity_threshold / 100
            )
        except Exception as e:
            return f"Error: {str(e)}"
    
    interface = gr.Interface(
        fn=process_text,
        inputs=[
            gr.Textbox(label="Input Text", placeholder="Enter text to enhance...", lines=10),
            gr.Slider(minimum=50, maximum=100, value=80, label="Minimum Semantic Similarity (%)")
        ],
        outputs=gr.Textbox(label="Enhanced Text", lines=10),
        title="RePhrase",
        description="Improve text quality while preserving original meaning"
    )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()
