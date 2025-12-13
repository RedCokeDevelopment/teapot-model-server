# TeaScout Model Server

This is the server component for hosting and serving TeaScout (The AI moderation tool used by Teapot) models via a RESTful API. It supports multiple model architectures through an adapter pattern.

## Installation

1.  **Prerequisites**: Ensure you have Python 3.8+ installed.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Model Weights**: Place your trained PyTorch model files (`.pt`) in the `models/` directory.

## Configuration (`models.json`)

The `models.json` file defines which models to load and how to configure them.

**Example:**

```json
{
    "teascout-v3": {
        "adapter": "teascout_v3",
        "path": "models/teascout_v3.pt",
        "config": {
            "embedding_dim": 128,
            "hidden_dim": 256,
            "output_dim": 1,
            "n_layers": 3,
            "dropout": 0.2,
            "max_len": 128
        },
        "description": "Model description here"
    }
}
```

## Usage

1.  **Start the Server**:
    ```bash
    python app.py
    ```
    The server will start on `http://0.0.0.0:5000`.

2.  **API Endpoints**:

    *   **List Available Models**
        *   **URL**: `/models`
        *   **Method**: `GET`
        *   **Response**:
            ```json
            {
                "teascout-v3": "Model description here",
            }
            ```

    *   **Run Inference**
        *   **URL**: `/predict/<model_name>` (e.g., `/predict/teascout-v3`)
        *   **Method**: `POST`
        *   **Body**:
            ```json
            {
                "text": "Text to analyze"
            }
            ```
        *   **Response**:
            ```json
            {
                "score": 0.0023,
            }
            ```

## Adding New Models (Adapters)

To support a new model architecture:

1.  Create a new Python file in the `adapters/` directory, Define a class named `Adapter` with the following structure:

    ```python
    class Adapter:
        def setup(self, model_path, config, device='cpu'):
            """
            Initialize the model and tokenizer.
            Args:
                model_path (str): Path to the model checkpoint.
                config (dict): Configuration dictionary from models.json.
                device (str): Device to load the model on ('cpu' or 'cuda').
            """
            # Load model and tokenizer here
            pass

        def inference(self, text):
            """
            Run inference on the input text.
            Args:
                text (str): Input text string.
            Returns:
                dict: A dictionary containing the results (e.g., label, score).
            """
            # Run inference here
            return {"label": "...", "score": ...}
    ```
2.  Change your model entry in `models.json` to reference the new adapter.
