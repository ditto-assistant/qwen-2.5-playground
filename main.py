from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
import time
from datetime import datetime

app = Flask(__name__)

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add file handler for logging
file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.addHandler(file_handler)

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    start_time = time.time()
    logger.info("Starting model and tokenizer loading process...")
    
    try:
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        logger.info(f"Loading tokenizer from {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Tokenizer loaded successfully")
        
        logger.info(f"Loading model from {model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        logger.info("Model loaded successfully")
        
        end_time = time.time()
        load_time = end_time - start_time
        logger.info(f"Total loading time: {load_time:.2f} seconds")
        
        # Log model information
        logger.info(f"Model device: {next(model.parameters()).device}")
        logger.info(f"Model dtype: {next(model.parameters()).dtype}")
        if torch.cuda.is_available():
            logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            logger.info(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        raise

@app.route('/')
def home():
    logger.info("Home page accessed")
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        start_time = time.time()
        prompt = request.json['prompt']
        logger.info(f"Received generation request with prompt length: {len(prompt)} characters")
        
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        logger.debug("Applying chat template")
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        logger.debug("Tokenizing input")
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        input_token_count = model_inputs.input_ids.shape[1]
        logger.info(f"Input token count: {input_token_count}")
        
        logger.info("Generating response")
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        output_token_count = len(generated_ids[0])
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        logger.info(f"Generation completed in {generation_time:.2f} seconds")
        logger.info(f"Output token count: {output_token_count}")
        
        if torch.cuda.is_available():
            logger.debug(f"GPU Memory after generation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        return jsonify({
            'response': response,
            'metadata': {
                'generation_time': f"{generation_time:.2f}s",
                'input_tokens': input_token_count,
                'output_tokens': output_token_count
            }
        })
        
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.before_request
def log_request_info():
    logger.debug('Headers: %s', request.headers)
    logger.debug('Body: %s', request.get_data())

@app.after_request
def log_response_info(response):
    logger.debug('Response: %s', response.get_data())
    return response

if __name__ == '__main__':
    print("\nInitializing application...")
    logger.info("Application startup")
    logger.info(f"Python version: {torch.__version__}")
    logger.info(f"Torch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    print("\nLoading model, please wait...")
    load_model()
    
    port = 5000
    logger.info(f"Starting server on port {port}")
    print(f"\n🚀 Server running at http://localhost:{port}")
    
    app.run(debug=True, port=port)
