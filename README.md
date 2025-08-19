# 🤖 AI Text Summarizer with FLAN-T5 + LoRA

A professional web application for intelligent text summarization using fine-tuned FLAN-T5 with LoRA (Low-Rank Adaptation) technique. This application provides an intuitive interface for generating high-quality summaries with customizable parameters and comprehensive analytics.

## ✨ Features

### 🎯 **Core Functionality**
- **Intelligent Summarization**: Advanced text summarization using fine-tuned FLAN-T5-Base
- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning for improved performance
- **Multi-language Support**: Arabic and English interface
- **Real-time Processing**: Fast inference with GPU acceleration support

### 🖥️ **Professional Interface**
- **Modern UI/UX**: Clean, responsive design with custom styling
- **Interactive Controls**: Adjustable summary parameters (length, compression ratio)
- **Progress Tracking**: Real-time progress indicators and status updates
- **Export Capabilities**: Download summaries as text files

### 📊 **Analytics & Insights**
- **Performance Metrics**: Generation time, compression ratios, word counts
- **Summary History**: Track and review previous summaries
- **Text Statistics**: Comprehensive analysis of input and output text
- **System Information**: GPU/CPU usage, memory consumption

### 🔧 **Technical Features**
- **Model Caching**: Optimized loading with Streamlit caching
- **Error Handling**: Comprehensive error management and user feedback
- **Device Compatibility**: Automatic GPU/CPU detection and optimization
- **Memory Management**: Efficient resource utilization

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
CUDA-compatible GPU (optional, for faster inference)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-text-summarizer.git
cd ai-text-summarizer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up the model**
   - Download your trained LoRA adapter from Google Colab
   - Extract `peft-dialogue-summary-ckpt.zip` to the project root
   - Ensure the following structure:
   ```
   ai-text-summarizer/
   ├── app.py
   ├── requirements.txt
   └── peft-dialogue-summary-ckpt/
       ├── adapter_config.json
       ├── adapter_model.safetensors
       ├── tokenizer.json
       └── ...
   ```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Access the app**
   - Open your browser and navigate to `http://localhost:8501`

## 📦 Model Setup

### From Google Colab

If you trained your model in Google Colab, follow these steps:

```python
# In your Colab notebook (after training)
# Save the model and tokenizer
peft_model_path = './peft-dialogue-summary-ckpt'
peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)

# Create zip file for download
!zip -r peft-dialogue-summary-ckpt.zip peft-dialogue-summary-ckpt/

# Download to your local machine
from google.colab import files
files.download("peft-dialogue-summary-ckpt.zip")
```

### From Hugging Face Hub

If you uploaded your model to Hugging Face:

1. Update the model path in `app.py`:
```python
model_source = "your-username/your-model-name"
```

2. The app will automatically download the model on first run

## 🛠️ Configuration

### Model Parameters

You can customize the summarization behavior through the sidebar:

- **Maximum Length**: Control the maximum length of generated summaries
- **Minimum Length**: Set the minimum acceptable summary length
- **Language**: Switch between Arabic and English interfaces

### Advanced Settings

For developers, you can modify these parameters in the code:

```python
# Generation parameters
length_penalty = 2.0      # Controls length preference
num_beams = 4            # Beam search width
early_stopping = True    # Stop when EOS token is generated
do_sample = False        # Use deterministic generation
```

## 📊 Project Structure

```
ai-text-summarizer/
├── 📄 app.py                               # Main Streamlit application
├── 📄 requirements.txt                     # Python dependencies
├── 📄 README.md                           # Project documentation
├── 📄 LICENSE                             # MIT License
├── 📁 peft-dialogue-summary-ckpt/         # Trained LoRA adapter
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   └── ...
├── 📁 assets/                             # Static assets (optional)
│   └── 🖼️ screenshots/
├── 📁 notebooks/                          # Training notebooks (optional)
│   └── 📓 training_notebook.ipynb
└── 📁 tests/                              # Unit tests (optional)
    └── 🧪 test_summarization.py
```

## 🎯 Usage Examples

### Basic Summarization

1. Launch the application
2. Enter or paste your text in the input area
3. Adjust summary parameters if needed
4. Click "Generate Summary"
5. View results with statistics and export options

### Supported Text Types

- ✅ **News Articles**: Current events, journalism
- ✅ **Academic Papers**: Research abstracts, papers
- ✅ **Blog Posts**: Technical blogs, tutorials
- ✅ **Reports**: Business reports, documentation
- ✅ **Stories**: Narrative content, literature
- ✅ **Dialogues**: Conversations, interviews

## 🔧 Technical Details

### Model Architecture

- **Base Model**: Google FLAN-T5-Base (248M parameters)
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Task**: Sequence-to-sequence text summarization
- **Training Data**: Custom dialogue and document datasets

### Performance Specifications

| Metric | Value |
|--------|--------|
| Model Size | ~250MB (base) + ~10MB (LoRA) |
| Inference Speed | ~2-5 seconds (GPU) / ~10-30 seconds (CPU) |
| Max Input Length | 512 tokens |
| Max Output Length | 150 tokens (configurable) |
| Memory Usage | ~2GB RAM (GPU) / ~4GB RAM (CPU) |

### System Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- 2GB free disk space

**Recommended:**
- Python 3.9+
- 8GB+ RAM
- NVIDIA GPU with 4GB+ VRAM
- 5GB+ free disk space

## 🛡️ Dependencies

### Core Libraries
```
streamlit>=1.28.0
transformers>=4.30.0
peft>=0.4.0
torch>=2.0.0
```

### Full Requirements
See `requirements.txt` for complete dependency list.

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Cloud Deployment

**Streamlit Cloud:**
1. Push your code to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy with one click

**Heroku:**
```bash
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

## 🧪 Testing

Run the test suite:
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v --cov=app
```

## 📈 Performance Optimization

### Speed Improvements
- **Model Caching**: Automatic model caching with Streamlit
- **GPU Acceleration**: CUDA support for faster inference
- **Batch Processing**: Efficient tokenization and generation

### Memory Optimization
- **Float16 Precision**: Reduced memory footprint on compatible hardware
- **Dynamic Loading**: Models loaded only when needed
- **Garbage Collection**: Automatic cleanup of unused tensors

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Add tests** for new functionality
5. **Submit a pull request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for the Transformers library and FLAN-T5 model
- **Microsoft** for the LoRA (PEFT) implementation
- **Streamlit** for the amazing web app framework
- **Google** for the original T5 architecture and FLAN-T5 model

## 📧 Contact & Support

- **Email**: your.email@example.com
- **GitHub Issues**: [Report bugs or request features](https://github.com/yourusername/ai-text-summarizer/issues)
- **Documentation**: [Full documentation](https://github.com/yourusername/ai-text-summarizer/wiki)

## 🔄 Version History

### v1.0.0 (Current)
- ✨ Initial release
- 🤖 FLAN-T5 + LoRA integration
- 🖥️ Professional Streamlit interface
- 📊 Analytics and statistics
- 🌐 Multi-language support

### Upcoming Features
- 🔄 Batch processing for multiple files
- 📊 Advanced analytics dashboard
- 🔐 User authentication system
- 🌍 Additional language support
- 📱 Mobile app version

---

<div align="center">

**Made with ❤️ for the AI community**

[⭐ Star this repo](https://github.com/yourusername/ai-text-summarizer) • [🐛 Report Bug](https://github.com/yourusername/ai-text-summarizer/issues) • [💡 Request Feature](https://github.com/yourusername/ai-text-summarizer/issues)

</div>