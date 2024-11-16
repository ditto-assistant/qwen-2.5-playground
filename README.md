# Qwen 2.5 Chat Interface 🤖

A simple API and web playground for the Qwen 2.5 language model, built with Flask and PyTorch. This application provides a clean, responsive chat interface for interacting with the Qwen 2.5 model running locally on your machine.

## 🌟 Features

- Real-time chat interface and local Flask API (http://localhost:5000) for hosting your own LLM Home Server.
- Response generation metrics
- Loading indicators and status updates
- Error handling and user feedback
- Responsive design
- GPU acceleration support (if you have CUDA)
- Comprehensive logging system

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA toolkit 11.8+ (optional, for GPU support)
- NVIDIA GPU with 4GB+ VRAM (optional, for GPU support)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <repo-directory>
   ```

2. **Create and activate a virtual environment (recommended)**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/MacOS
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyTorch with CUDA support (if using GPU)**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Running the Application

1. **Start the Flask server**
   ```bash
   python main.py
   ```

2. **Access the web interface**
   - Open your browser and navigate to `http://localhost:5000`
   - The model will take a few moments to load initially

## 🔧 Configuration

### Model Information

This application uses the Qwen 2.5 0.5B Instruct model:
- Model Name: `Qwen/Qwen2.5-0.5B-Instruct`
- [Model on Hugging Face](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- Parameters: 0.5B
- Context Length: Up to 32,768 tokens

### Environment Variables

No environment variables are required for basic operation. The application uses default settings:
- Port: 5000
- Debug Mode: Enabled
- Log Level: INFO

## 📝 Logging

The application maintains detailed logs in `app.log`, including:
- Model loading progress
- Request/response information
- Generation metrics
- Error tracking
- GPU memory usage (when applicable)

## 💻 System Requirements

### Minimum Requirements
- 8GB RAM
- 2GB free disk space
- Modern web browser

### Recommended Requirements
- 16GB RAM
- NVIDIA GPU with 4GB+ VRAM
- CUDA 11.8 or newer
- 5GB free disk space

## 🛠 Development

### Project Structure
```
.
├── main.py              # Flask application and model handling
├── templates/
│   └── index.html      # Web interface template
├── requirements.txt     # Python dependencies
├── app.log             # Application logs
└── README.md           # Documentation
```

### Adding Features

1. Backend modifications go in `main.py`
2. Frontend changes go in `templates/index.html`
3. New dependencies should be added to `requirements.txt`

## ⚠️ Known Issues

- First request might be slow due to model initialization
- CPU-only mode is significantly slower than GPU mode
- Large responses might take several seconds to generate

## 🙏 Acknowledgments

- [Flask](https://flask.palletsprojects.com/) for the web framework
- [TailwindCSS](https://tailwindcss.com/) for the styling
- [Hugging Face](https://huggingface.co/) for model hosting and transformers library

## 📚 Model Credits

This project uses the Qwen2.5 model. For detailed evaluation results and performance metrics, please visit the [official Qwen2.5 blog](https://qwenlm.github.io/blog/qwen2.5/).

If you use this model in your work, please cite:

```bibtex
@misc{qwen2.5,
    title = {Qwen2.5: A Party of Foundation Models},
    url = {https://qwenlm.github.io/blog/qwen2.5/},
    author = {Qwen Team},
    month = {September},
    year = {2024}
}

@article{qwen2,
    title = {Qwen2 Technical Report}, 
    author = {An Yang and Baosong Yang and Binyuan Hui and Bo Zheng and Bowen Yu and Chang Zhou 
              and Chengpeng Li and Chengyuan Li and Dayiheng Liu and Fei Huang and Guanting Dong 
              and Haoran Wei and Huan Lin and Jialong Tang and Jialin Wang and Jian Yang 
              and Jianhong Tu and Jianwei Zhang and Jianxin Ma and Jin Xu and Jingren Zhou 
              and Jinze Bai and Jinzheng He and Junyang Lin and Kai Dang and Keming Lu 
              and Keqin Chen and Kexin Yang and Mei Li and Mingfeng Xue and Na Ni and Pei Zhang 
              and Peng Wang and Ru Peng and Rui Men and Ruize Gao and Runji Lin and Shijie Wang 
              and Shuai Bai and Sinan Tan and Tianhang Zhu and Tianhao Li and Tianyu Liu 
              and Wenbin Ge and Xiaodong Deng and Xiaohuan Zhou and Xingzhang Ren and Xinyu Zhang 
              and Xipin Wei and Xuancheng Ren and Yang Fan and Yang Yao and Yichang Zhang 
              and Yu Wan and Yunfei Chu and Yuqiong Liu and Zeyu Cui and Zhenru Zhang and Zhihao Fan},
    journal = {arXiv preprint arXiv:2407.10671},
    year = {2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request