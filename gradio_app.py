import asyncio
import logging
import os
from typing import Tuple, Optional, Dict, Any

import gradio as gr
import httpx
from models import FFmpegRequest, ModelProvider


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FFmpegGradioInterface:
    """Gradio interface for FFmpeg natural language processing"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def generate_command(
        self,
        query: str,
        model_provider: str,
        additional_context: str = "",
        input_file: str = "",
        output_file: str = ""
    ) -> Tuple[str, str, str, str, str]:
        """Generate FFmpeg command from natural language query
        
        Returns:
            Tuple of (command, explanation, model_used, confidence, warnings)
        """
        
        if not query.strip():
            return "", "Please enter a query", "", "", ""
        
        try:
            # Prepare request
            request_data = {
                "query": query.strip(),
                "model_provider": model_provider,
                "additional_context": additional_context.strip() if additional_context else None,
                "input_file": input_file.strip() if input_file else None,
                "output_file": output_file.strip() if output_file else None
            }
            
            # Remove None values
            request_data = {k: v for k, v in request_data.items() if v is not None}
            
            # Make API call
            response = await self.client.post(
                f"{self.api_base_url}/generate",
                json=request_data
            )
            
            if response.status_code == 200:
                data = response.json()
                
                command = data.get("command", "")
                explanation = data.get("explanation", "")
                model_used = data.get("model_used", "")
                confidence = f"{data.get('confidence', 0):.2%}"
                warnings = "\n".join(data.get("warnings", [])) if data.get("warnings") else "None"
                
                return command, explanation, model_used, confidence, warnings
            
            else:
                error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                error_msg = error_data.get("detail", f"HTTP {response.status_code}")
                return "", f"Error: {error_msg}", "", "", ""
                
        except httpx.TimeoutException:
            return "", "Error: Request timed out. The AI service may be busy.", "", "", ""
        except httpx.ConnectError:
            return "", "Error: Cannot connect to the API server. Make sure it's running.", "", "", ""
        except Exception as e:
            logger.error(f"Error generating command: {e}")
            return "", f"Error: {str(e)}", "", "", ""
    
    async def validate_query(self, query: str) -> str:
        """Validate query and return analysis"""
        if not query.strip():
            return "Please enter a query to validate"
        
        try:
            response = await self.client.post(
                f"{self.api_base_url}/validate",
                json={"query": query.strip(), "model_provider": "auto"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("valid"):
                    complexity = data.get("complexity_score", 0)
                    suggested_models = ", ".join(data.get("suggested_models", []))
                    word_count = data.get("word_count", 0)
                    
                    return f"""✅ Query is valid
📊 Complexity Score: {complexity:.2f}
🤖 Suggested Models: {suggested_models}
📝 Word Count: {word_count}"""
                else:
                    return f"❌ Query validation failed: {data.get('error', 'Unknown error')}"
            else:
                return f"❌ Validation error: HTTP {response.status_code}"
                
        except Exception as e:
            return f"❌ Validation error: {str(e)}"
    
    async def check_health(self) -> str:
        """Check API health status"""
        try:
            response = await self.client.get(f"{self.api_base_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                status = data.get("status", "unknown")
                models = data.get("models_available", {})
                ffmpeg = data.get("ffmpeg_available", False)
                
                status_emoji = {
                    "healthy": "🟢",
                    "degraded": "🟡", 
                    "unhealthy": "🔴"
                }.get(status, "⚪")
                
                models_status = "\n".join([
                    f"  {name}: {'✅' if available else '❌'}"
                    for name, available in models.items()
                ])
                
                return f"""{status_emoji} Service Status: {status.upper()}

🤖 AI Models:
{models_status}

🎬 FFmpeg: {'✅ Available' if ffmpeg else '❌ Not Found'}"""
            else:
                return f"❌ Health check failed: HTTP {response.status_code}"
                
        except Exception as e:
            return f"❌ Cannot connect to service: {str(e)}"
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface"""
        
        with gr.Blocks(
            title="FSPEAK - FFmpeg Natural Language Interface",
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            .command-output {
                font-family: 'Courier New', monospace;
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #ddd;
            }
            """
        ) as interface:
            
            gr.Markdown(
                """
                # 🎬 FSPEAK - FFmpeg Natural Language Interface
                
                Convert natural language descriptions into optimized FFmpeg commands using AI.
                Simply describe what you want to do with your media files!
                
                **Examples:**
                - "Convert video.mp4 to audio.mp3"
                - "Crop the first 10 seconds of my_video.avi"
                - "Add subtitles.srt to movie.mp4 and compress to 720p"
                - "Extract audio from video and convert to WAV format"
                """
            )
            
            with gr.Tab("🎯 Generate Command"):
                with gr.Row():
                    with gr.Column(scale=2):
                        query_input = gr.Textbox(
                            label="📝 Describe what you want to do",
                            placeholder="e.g., convert my_video.mp4 to audio.mp3 with high quality",
                            lines=3,
                            max_lines=5
                        )
                        
                        with gr.Row():
                            model_dropdown = gr.Dropdown(
                                choices=["auto", "openai", "ollama", "gemini"],
                                value="auto",
                                label="🤖 AI Model",
                                info="Choose AI model or let the system decide automatically"
                            )
                            
                        with gr.Accordion("⚙️ Advanced Options", open=False):
                            additional_context = gr.Textbox(
                                label="📋 Additional Context",
                                placeholder="Any specific requirements or constraints...",
                                lines=2
                            )
                            
                            with gr.Row():
                                input_file = gr.Textbox(
                                    label="📁 Input File (optional)",
                                    placeholder="input.mp4"
                                )
                                output_file = gr.Textbox(
                                    label="💾 Output File (optional)",
                                    placeholder="output.mp3"
                                )
                        
                        with gr.Row():
                            generate_btn = gr.Button(
                                "🚀 Generate FFmpeg Command",
                                variant="primary",
                                size="lg"
                            )
                            validate_btn = gr.Button(
                                "✅ Validate Query",
                                variant="secondary"
                            )
                    
                    with gr.Column(scale=1):
                        health_display = gr.Textbox(
                            label="🏥 Service Status",
                            value="Click 'Check Status' to update",
                            lines=8,
                            interactive=False
                        )
                        health_btn = gr.Button("🔄 Check Status")
                
                gr.Markdown("## 📤 Generated Command")
                
                command_output = gr.Textbox(
                    label="💻 FFmpeg Command",
                    lines=3,
                    max_lines=5,
                    show_copy_button=True,
                    elem_classes=["command-output"]
                )
                
                with gr.Row():
                    with gr.Column():
                        explanation_output = gr.Textbox(
                            label="📖 Explanation",
                            lines=4,
                            interactive=False
                        )
                    
                    with gr.Column():
                        with gr.Row():
                            model_used_output = gr.Textbox(
                                label="🤖 Model Used",
                                interactive=False
                            )
                            confidence_output = gr.Textbox(
                                label="🎯 Confidence",
                                interactive=False
                            )
                        
                        warnings_output = gr.Textbox(
                            label="⚠️ Warnings",
                            lines=3,
                            interactive=False
                        )
                
                validation_output = gr.Textbox(
                    label="✅ Query Validation",
                    lines=4,
                    interactive=False,
                    visible=False
                )
            
            with gr.Tab("📚 Examples & Help"):
                gr.Markdown(
                    """
                    ## 🎯 Common Use Cases
                    
                    ### 🔄 Format Conversion
                    - "Convert MP4 to MP3"
                    - "Change AVI to MOV format"
                    - "Convert video to GIF"
                    
                    ### ✂️ Editing Operations
                    - "Cut first 30 seconds from video"
                    - "Extract 2 minutes starting from 1:30"
                    - "Trim video from 00:10 to 02:45"
                    
                    ### 🎨 Quality & Compression
                    - "Compress video to 720p"
                    - "Reduce file size while maintaining quality"
                    - "Convert to high quality MP3"
                    
                    ### 🎵 Audio Operations
                    - "Extract audio from video"
                    - "Remove audio from video"
                    - "Adjust audio volume to 150%"
                    
                    ### 📝 Subtitles & Overlays
                    - "Add subtitles from SRT file"
                    - "Burn subtitles into video"
                    - "Add watermark to video"
                    
                    ## 🤖 AI Model Selection
                    
                    - **Auto**: Let the system choose the best model based on query complexity
                    - **OpenAI**: Best for complex operations and high accuracy
                    - **Ollama**: Local processing, good for privacy and offline use
                    - **Gemini**: Excellent for media-specific operations and context understanding
                    
                    ## 💡 Tips for Better Results
                    
                    1. **Be specific**: Include file names and desired formats
                    2. **Mention quality**: Specify if you want high quality, compression, etc.
                    3. **Include timing**: For cuts/trims, specify exact timestamps
                    4. **Add context**: Use the additional context field for special requirements
                    
                    ## ⚠️ Important Notes
                    
                    - Always review generated commands before running them
                    - Test with sample files first for complex operations
                    - The system assumes FFmpeg is installed at `C:\\ffmpeg\\bin\\ffmpeg.exe`
                    - Some operations may take significant processing time
                    """
                )
            
            # Event handlers
            def sync_generate_command(*args):
                return asyncio.run(self.generate_command(*args))
            
            def sync_validate_query(query):
                result = asyncio.run(self.validate_query(query))
                return gr.update(value=result, visible=True)
            
            def sync_check_health():
                return asyncio.run(self.check_health())
            
            # Wire up the interface
            generate_btn.click(
                fn=sync_generate_command,
                inputs=[
                    query_input,
                    model_dropdown,
                    additional_context,
                    input_file,
                    output_file
                ],
                outputs=[
                    command_output,
                    explanation_output,
                    model_used_output,
                    confidence_output,
                    warnings_output
                ]
            )
            
            validate_btn.click(
                fn=sync_validate_query,
                inputs=[query_input],
                outputs=[validation_output]
            )
            
            health_btn.click(
                fn=sync_check_health,
                outputs=[health_display]
            )
            
            # Auto-hide validation output when generating
            generate_btn.click(
                fn=lambda: gr.update(visible=False),
                outputs=[validation_output]
            )
        
        return interface
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


def create_gradio_app(api_base_url: str = "http://localhost:8000") -> gr.Blocks:
    """Create and return the Gradio interface"""
    interface = FFmpegGradioInterface(api_base_url)
    return interface.create_interface()


if __name__ == "__main__":
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Get API URL from environment or use default
    api_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    
    # Create and launch the interface
    app = create_gradio_app(api_url)
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )