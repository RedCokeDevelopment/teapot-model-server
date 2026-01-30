import os
import json
import importlib.util
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

# Configuration
MODELS_CONFIG_FILE = 'models/models.json'
ADAPTERS_DIR = 'adapters'
MODELS_DIR = 'models'

# Global storage for loaded models
loaded_models = {}

def load_adapter(adapter_name):
    """Dynamically load an adapter module."""
    adapter_path = os.path.join(ADAPTERS_DIR, f"{adapter_name}.py")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter {adapter_name} not found at {adapter_path}")
    
    spec = importlib.util.spec_from_file_location(adapter_name, adapter_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if not hasattr(module, 'Adapter'):
        raise AttributeError(f"Adapter module {adapter_name} must define an 'Adapter' class")
        
    return module.Adapter

def init_models():
    """Initialize all models defined in models.json."""
    global loaded_models
    
    if not os.path.exists(MODELS_CONFIG_FILE):
        print(f"Warning: {MODELS_CONFIG_FILE} not found.")
        return

    with open(MODELS_CONFIG_FILE, 'r') as f:
        config = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    for model_name, model_config in config.items():
        try:
            print(f"Loading model: {model_name}...")
            adapter_name = model_config.get('adapter')
            model_path = model_config.get('path')
            model_params = model_config.get('config', {})

            if not adapter_name or not model_path:
                print(f"Skipping {model_name}: Missing adapter or path configuration.")
                continue

            # Load Adapter Class
            AdapterClass = load_adapter(adapter_name)
            adapter_instance = AdapterClass()
            
            # Setup Model
            # Ensure model path is absolute or relative to CWD
            if not os.path.isabs(model_path):
                model_path = os.path.abspath(model_path)
                
            adapter_instance.setup(model_path, model_params, device=device)
            
            loaded_models[model_name] = {
                "instance": adapter_instance,
                "info": model_config
            }
            print(f"Successfully loaded {model_name}")
            
        except Exception as e:
            print(f"Failed to load {model_name}: {str(e)}")

@app.route('/models', methods=['GET'])
def list_models():
    """List available models."""
    return jsonify({
        name: {
            'description': info['info'].get('description', 'No description'),
            'optimized_threshold': info['info'].get('Optimized_Threshold', None)
        }
        for name, info in loaded_models.items()
    })

@app.route('/inference/<model_name>', methods=['POST'])
def inference(model_name):
    """Run inference on a specific model."""
    if model_name not in loaded_models:
        return jsonify({"error": f"Model '{model_name}' not found"}), 404

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    text = data['text']
    model_instance = loaded_models[model_name]['instance']

    try:
        result = model_instance.inference(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Model Server...")
    init_models()
    app.run(host='0.0.0.0', port=5000)
