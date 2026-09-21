from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Reuired API keys
    deepgram_api_key: str
    groq_api_key: str
    elevenlabs_api_key: str

    # LLM
    llm_model: str = "llama-3.1-8b-instant"
    llm_max_tokens: int = 512
    llm_temperature: float = 0.7

    # Speech To Text
    stt_model: str = "nova-2"
    stt_sample_rate: int = 16000
    stt_language: str = "en"

    # Text To Speech
    tts_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    tts_model: str = "eleven_turbo_v2_5"

    # VAoice Activity Detection
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 500

    # Pipeline
    queue_maxsize: int = 64

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8765

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()