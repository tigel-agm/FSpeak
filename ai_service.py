import asyncio
import re
import logging
from typing import Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod

import httpx
from models import ModelProvider, ModelConfig, FFmpegRequest, FFmpegResponse


logger = logging.getLogger(__name__)


class AIModelInterface(ABC):
    """Abstract interface for AI model providers"""
    
    @abstractmethod
    async def generate_command(self, request: FFmpegRequest) -> Tuple[str, str, float]:
        """Generate FFmpeg command from natural language request
        
        Returns:
            Tuple of (command, explanation, confidence_score)
        """
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the model provider is available"""
        pass


class OpenAIProvider(AIModelInterface):
    """OpenAI GPT model provider"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = None
        if config.openai_api_key:
            try:
                import openai
                self.client = openai.AsyncOpenAI(api_key=config.openai_api_key)
            except ImportError:
                logger.warning("OpenAI package not installed")
    
    async def generate_command(self, request: FFmpegRequest) -> Tuple[str, str, float]:
        if not self.client:
            raise ValueError("OpenAI client not configured")
        
        system_prompt = """
You are an expert FFmpeg command generator. Convert natural language requests into valid FFmpeg commands.

Rules:
1. Generate ONLY the FFmpeg command, no explanations in the command itself
2. Use the most efficient and commonly supported options
3. Assume FFmpeg is installed at C:\\ffmpeg\\bin\\ffmpeg.exe on Windows
4. Include proper input/output file handling
5. Use appropriate codecs and formats
6. Consider file size optimization when relevant

Respond with JSON format:
{
  "command": "ffmpeg command here",
  "explanation": "Clear explanation of what this command does",
  "confidence": 0.95
}
"""
        
        user_prompt = f"Convert this request to FFmpeg command: {request.query}"
        if request.additional_context:
            user_prompt += f"\nAdditional context: {request.additional_context}"
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("OpenAI returned empty response")
            return self._parse_response(content)
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def is_available(self) -> bool:
        return self.client is not None
    
    def _parse_response(self, content: str) -> Tuple[str, str, float]:
        """Parse JSON response from OpenAI"""
        import json
        try:
            data = json.loads(content)
            return (
                data.get("command", ""),
                data.get("explanation", ""),
                float(data.get("confidence", 0.8))
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback parsing if JSON fails
            lines = content.strip().split('\n')
            command = ""
            explanation = ""
            
            for line in lines:
                if line.startswith('ffmpeg'):
                    command = line
                elif line and not command:
                    explanation += line + " "
            
            return command.strip(), explanation.strip(), 0.7


class OllamaProvider(AIModelInterface):
    """Ollama local model provider"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.base_url = config.ollama_base_url
    
    async def generate_command(self, request: FFmpegRequest) -> Tuple[str, str, float]:
        prompt = f"""
You are an FFmpeg expert. Convert this natural language request to a valid FFmpeg command:

Request: {request.query}
{f'Context: {request.additional_context}' if request.additional_context else ''}

Rules:
- Generate only the FFmpeg command
- Use C:\\ffmpeg\\bin\\ffmpeg.exe as the executable path
- Be concise and efficient
- Include brief explanation after the command

Format:
Command: [ffmpeg command]
Explanation: [what it does]
"""
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.config.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "top_p": 0.9
                        }
                    }
                )
                response.raise_for_status()
                data = response.json()
                return self._parse_ollama_response(data.get("response", ""))
                
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except:
            return False
    
    def _parse_ollama_response(self, content: str) -> Tuple[str, str, float]:
        """Parse Ollama response"""
        lines = content.strip().split('\n')
        command = ""
        explanation = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('Command:'):
                command = line.replace('Command:', '').strip()
            elif line.startswith('Explanation:'):
                explanation = line.replace('Explanation:', '').strip()
            elif 'ffmpeg' in line.lower() and not command:
                command = line
        
        # If no structured response, try to extract ffmpeg command
        if not command:
            for line in lines:
                if line.strip().startswith('ffmpeg') or 'C:\\ffmpeg' in line:
                    command = line.strip()
                    break
        
        if not explanation:
            explanation = "FFmpeg command generated by local model"
        
        return command, explanation, 0.75


class GeminiProvider(AIModelInterface):
    """Google Gemini model provider"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = None
        if config.gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=config.gemini_api_key)
                self.client = genai.GenerativeModel(config.gemini_model)
            except ImportError:
                logger.warning("Google GenerativeAI package not installed")
    
    async def generate_command(self, request: FFmpegRequest) -> Tuple[str, str, float]:
        if not self.client:
            raise ValueError("Gemini client not configured")
        
        prompt = f"""
As an FFmpeg expert, convert this natural language request into a precise FFmpeg command:

Request: {request.query}
{f'Additional context: {request.additional_context}' if request.additional_context else ''}

Requirements:
1. Use C:\\ffmpeg\\bin\\ffmpeg.exe as the executable
2. Generate efficient, working commands
3. Consider media format compatibility
4. Optimize for quality and file size when appropriate

Provide:
1. The exact FFmpeg command
2. A clear explanation of what it does
3. Your confidence level (0.0-1.0)

Format your response as:
COMMAND: [ffmpeg command]
EXPLANATION: [explanation]
CONFIDENCE: [0.0-1.0]
"""
        
        try:
            import google.generativeai as genai
            generation_config = genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=500
            )
            response = await asyncio.to_thread(
                self.client.generate_content,
                prompt,
                generation_config=generation_config
            )
            return self._parse_gemini_response(response.text)
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    async def is_available(self) -> bool:
        return self.client is not None
    
    def _parse_gemini_response(self, content: str) -> Tuple[str, str, float]:
        """Parse Gemini response"""
        command = ""
        explanation = ""
        confidence = 0.8
        
        lines = content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('COMMAND:'):
                command = line.replace('COMMAND:', '').strip()
            elif line.startswith('EXPLANATION:'):
                explanation = line.replace('EXPLANATION:', '').strip()
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.replace('CONFIDENCE:', '').strip())
                except ValueError:
                    pass
        
        # Fallback extraction
        if not command:
            for line in lines:
                if 'ffmpeg' in line.lower():
                    command = line.strip()
                    break
        
        return command, explanation, confidence


class AIService:
    """Main AI service that routes requests to appropriate model providers"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.providers = {
            ModelProvider.OPENAI: OpenAIProvider(config),
            ModelProvider.OLLAMA: OllamaProvider(config),
            ModelProvider.GEMINI: GeminiProvider(config)
        }
    
    async def generate_ffmpeg_command(self, request: FFmpegRequest) -> FFmpegResponse:
        """Generate FFmpeg command using the best available model"""
        
        # Determine which model to use
        if request.model_provider != ModelProvider.AUTO:
            provider = self.providers[request.model_provider]
            if not await provider.is_available():
                raise ValueError(f"Requested model provider {request.model_provider} is not available")
        else:
            provider = await self._select_best_provider(request)
        
        # Generate command
        try:
            command, explanation, confidence = await provider.generate_command(request)
            
            # Validate and clean the command
            command = self._validate_command(command)
            warnings = self._analyze_command(command)
            
            return FFmpegResponse(
                command=command,
                explanation=explanation,
                model_used=self._get_provider_type(provider),
                confidence=confidence,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Error generating command: {e}")
            raise
    
    async def _select_best_provider(self, request: FFmpegRequest) -> AIModelInterface:
        """Select the best model provider based on request complexity and availability"""
        
        # Check availability of providers
        available_providers = []
        for provider_type, provider in self.providers.items():
            if await provider.is_available():
                available_providers.append((provider_type, provider))
        
        if not available_providers:
            raise ValueError("No AI model providers are available")
        
        # Analyze request complexity
        complexity = self._analyze_complexity(request.query)
        
        # Routing logic
        if self.config.prefer_local and any(p[0] == ModelProvider.OLLAMA for p in available_providers):
            return next(p[1] for p in available_providers if p[0] == ModelProvider.OLLAMA)
        
        if complexity > self.config.complexity_threshold:
            # High complexity - prefer OpenAI or Gemini
            for provider_type in [ModelProvider.OPENAI, ModelProvider.GEMINI, ModelProvider.OLLAMA]:
                provider = next((p[1] for p in available_providers if p[0] == provider_type), None)
                if provider:
                    return provider
        else:
            # Low complexity - any provider is fine, prefer local
            for provider_type in [ModelProvider.OLLAMA, ModelProvider.OPENAI, ModelProvider.GEMINI]:
                provider = next((p[1] for p in available_providers if p[0] == provider_type), None)
                if provider:
                    return provider
        
        # Fallback to first available
        return available_providers[0][1]
    
    def _analyze_complexity(self, query: str) -> float:
        """Analyze query complexity to determine best model"""
        complexity_indicators = [
            r'\b(filter|complex|multiple|batch|script)\b',
            r'\b(overlay|watermark|subtitle|caption)\b',
            r'\b(custom|advanced|professional)\b',
            r'\b(and|then|also|plus|with)\b',  # Multiple operations
            r'\d+\s*(fps|kbps|mbps|resolution)',  # Specific technical parameters
        ]
        
        score = 0.0
        query_lower = query.lower()
        
        for pattern in complexity_indicators:
            if re.search(pattern, query_lower):
                score += 0.2
        
        # Length-based complexity
        if len(query.split()) > 10:
            score += 0.1
        if len(query.split()) > 20:
            score += 0.2
        
        return min(score, 1.0)
    
    def _validate_command(self, command: str) -> str:
        """Validate and clean the FFmpeg command"""
        if not command:
            raise ValueError("Empty command generated")
        
        # Ensure it starts with ffmpeg
        if not command.strip().startswith(('ffmpeg', 'C:\\ffmpeg')):
            if 'ffmpeg' in command:
                # Extract the ffmpeg part
                parts = command.split()
                ffmpeg_idx = next(i for i, part in enumerate(parts) if 'ffmpeg' in part)
                command = ' '.join(parts[ffmpeg_idx:])
            else:
                command = f"C:\\ffmpeg\\bin\\ffmpeg.exe {command}"
        
        # Replace generic ffmpeg with full path
        if command.startswith('ffmpeg '):
            command = command.replace('ffmpeg ', 'C:\\ffmpeg\\bin\\ffmpeg.exe ', 1)
        
        return command.strip()
    
    def _analyze_command(self, command: str) -> list[str]:
        """Analyze command for potential warnings"""
        warnings = []
        
        if '-y' not in command and '-n' not in command:
            warnings.append("Command may prompt for file overwrite confirmation. Consider adding -y flag.")
        
        if re.search(r'-crf\s+([0-9]+)', command):
            crf_match = re.search(r'-crf\s+([0-9]+)', command)
            if crf_match:
                crf_value = int(crf_match.group(1))
                if crf_value < 18:
                    warnings.append("Very high quality setting may result in large file sizes.")
                elif crf_value > 28:
                    warnings.append("Low quality setting may result in visible compression artifacts.")
        
        if '-filter_complex' in command:
            warnings.append("Complex filter operations may take significant processing time.")
        
        return warnings
    
    def _get_provider_type(self, provider: AIModelInterface) -> ModelProvider:
        """Get the provider type from provider instance"""
        if isinstance(provider, OpenAIProvider):
            return ModelProvider.OPENAI
        elif isinstance(provider, OllamaProvider):
            return ModelProvider.OLLAMA
        elif isinstance(provider, GeminiProvider):
            return ModelProvider.GEMINI
        else:
            return ModelProvider.AUTO
    
    async def get_model_status(self) -> Dict[str, bool]:
        """Get availability status of all model providers"""
        status = {}
        for provider_type, provider in self.providers.items():
            status[provider_type.value] = await provider.is_available()
        return status