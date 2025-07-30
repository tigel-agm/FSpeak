# 🎬 FSPEAK - FFmpeg Natural Language Interface

A lightweight, fast, and accurate natural language interface for FFmpeg that converts plain English descriptions into optimized FFmpeg commands using AI.

https://hmvlagent.com/fspeak.html

## ✨ Features

- **Natural Language Processing**: Describe what you want in plain English
- **Multi-AI Support**: OpenAI, Ollama (local), and Gemini integration
- **Smart Model Routing**: Automatic model selection based on query complexity
- **FastAPI Backend**: High-performance API with automatic documentation
- **Gradio Frontend**: Intuitive web interface for easy interaction
- **Command Validation**: Built-in FFmpeg command validation and optimization
- **Real-time Health Monitoring**: Check AI models and FFmpeg availability

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- FFmpeg installed at `C:\ffmpeg` (already installed as mentioned)
- At least one AI model API key (OpenAI, Gemini) or Ollama running locally

### Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   ```bash
   copy .env.example .env
   ```
   Edit `.env` and add your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Start the backend**:
   ```bash
   python main.py
   ```

5. **Start the frontend** (in a new terminal):
   ```bash
   python gradio_app.py
   ```

6. **Open your browser** and go to `http://localhost:7860`

## 🎯 Usage Examples

### Basic Conversions
- "Convert video.mp4 to audio.mp3"
- "Change my_movie.avi to MP4 format"
- "Convert video to high quality GIF"

### Video Editing
- "Cut the first 30 seconds from video.mp4"
- "Extract 2 minutes starting from 1:30"
- "Trim video from 00:10 to 02:45"

### Quality & Compression
- "Compress video.mp4 to 720p"
- "Reduce file size while maintaining quality"
- "Convert to high quality MP3 at 320kbps"

### Audio Operations
- "Extract audio from movie.mp4"
- "Remove audio track from video"
- "Increase audio volume by 50%"

### Advanced Operations
- "Add subtitles.srt to movie.mp4 and compress to 1080p"
- "Merge video.mp4 and audio.wav into output.mp4"
- "Add watermark logo.png to video at top-right corner"

## 🤖 AI Models

### OpenAI (Recommended)
- **Best for**: Complex operations, high accuracy
- **Setup**: Get API key from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Cost**: Pay-per-use

### Gemini
- **Best for**: Media-specific operations, context understanding
- **Setup**: Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Cost**: Generous free tier

### Ollama (Local)
- **Best for**: Privacy, offline use, no API costs
- **Setup**: Install [Ollama](https://ollama.ai/) and run `ollama pull llama3.2:3b`
- **Cost**: Free (uses local compute)

## 🔧 API Endpoints

### Generate Command
```http
POST /generate
Content-Type: application/json

{
  "query": "convert video.mp4 to audio.mp3",
  "model_provider": "auto",
  "additional_context": "high quality audio",
  "input_file": "video.mp4",
  "output_file": "audio.mp3"
}
```

### Validate Query
```http
POST /validate
Content-Type: application/json

{
  "query": "convert video to audio",
  "model_provider": "auto"
}
```

### Health Check
```http
GET /health
```

### Model Status
```http
GET /models
```

## 📁 Project Structure

```
FSPEAK/
├── main.py              # FastAPI backend server
├── gradio_app.py        # Gradio frontend interface
├── models.py            # Pydantic data models
├── ai_service.py        # AI model integration
├── requirements.txt     # Python dependencies
├── .env.example         # Environment configuration template
└── README.md           # This file
```

## ⚙️ Configuration

### Environment Variables

Key configuration options in `.env`:

```env
# API Keys
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here

# Model Settings
DEFAULT_MODEL_PROVIDER=auto
MODEL_PRIORITY=openai,gemini,ollama

# FFmpeg Path
FFMPEG_PATH=C:\ffmpeg\bin\ffmpeg.exe

# Server Settings
FAST_API_PORT=8000
GRADIO_PORT=7860
```

### Model Selection Logic

1. **Auto Mode** (default): Analyzes query complexity and selects best available model
2. **Manual Override**: Force specific model via dropdown or API parameter
3. **Fallback Chain**: If preferred model fails, tries next available model

## 🔍 Troubleshooting

### Common Issues

**"Cannot connect to API server"**
- Ensure the FastAPI backend is running (`python main.py`)
- Check if port 8000 is available
- Verify API_BASE_URL in `.env`

**"No AI models available"**
- Add at least one API key to `.env`
- For Ollama: ensure it's running (`ollama serve`)
- Check model status at `http://localhost:8000/health`

**"FFmpeg not found"**
- Verify FFmpeg is installed at `C:\ffmpeg\bin\ffmpeg.exe`
- Update FFMPEG_PATH in `.env` if installed elsewhere
- Test FFmpeg: `C:\ffmpeg\bin\ffmpeg.exe -version`

**"Invalid FFmpeg command generated"**
- Try a different AI model
- Add more context to your query
- Check the warnings in the response

### Debug Mode

Enable detailed logging:
```env
LOG_LEVEL=DEBUG
DETAILED_ERRORS=true
```

## 🎨 Customization

### Adding Custom Models

1. Extend `AIModelInterface` in `ai_service.py`
2. Add model configuration to `models.py`
3. Update model routing logic

### Custom FFmpeg Validation

Modify the `validate_ffmpeg_command` method in `ai_service.py` to add custom validation rules.

### UI Customization

Edit `gradio_app.py` to customize the interface:
- Change themes and styling
- Add new input fields
- Modify layout and components

## 📊 Performance Tips

1. **Use Auto Mode**: Let the system choose the best model for each query
2. **Cache Results**: Enable caching for repeated queries
3. **Batch Processing**: Process multiple files with similar operations
4. **Local Models**: Use Ollama for privacy and reduced latency
5. **Specific Queries**: More specific descriptions yield better results

## 🔒 Security Notes

- API keys are stored in environment variables (not in code)
- No authentication required (as specified)
- CORS enabled for development (disable in production)
- Input validation prevents malicious commands
- Rate limiting available (configure in `.env`)

## 🤝 Contributing

This is a focused implementation without tests or extensive documentation. For improvements:

1. Test generated commands before running
2. Report issues with specific queries and expected outputs
3. Suggest model routing improvements
4. Share common use cases for better prompt engineering

## 📄 License

This project is provided as-is for educational and practical use.

## 🙏 Acknowledgments

- [FFmpeg](https://ffmpeg.org/) - The complete, cross-platform solution to record, convert and stream audio and video
- [FastAPI](https://fastapi.tiangolo.com/) - Modern, fast web framework for building APIs
- [Gradio](https://gradio.app/) - Build and share delightful machine learning apps
- [OpenAI](https://openai.com/) - Advanced AI language models
- [Google Gemini](https://ai.google.dev/) - Multimodal AI capabilities
- [Ollama](https://ollama.ai/) - Local AI model inference
