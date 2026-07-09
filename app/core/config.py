import os
from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class AppSettings(BaseModel):
    name: str = "PersonalAssistant"
    cors_origins: list[str] = [
        "http://localhost:5173",
        "https://muhammadumerkhaninfo.vercel.app",
        "https://umerr.vercel.app"
    ]
    resume_path: Path = PROJECT_ROOT / "assets" / "Muhammad_Umer_Khan_AI_Resume.pdf"
    model_name: str = "openai/gpt-oss-120b"
    guard_model_name: str = "llama-3.1-8b-instant"
    embedding_model: str = "BAAI/bge-base-en-v1.5"

class GroqSettings(BaseModel):
    api_key: SecretStr
    fallback_api_key: SecretStr | None = None


class QdrantSettings(BaseModel):
    api_key: SecretStr
    endpoint: str
    cluster_id: str
    collection_name: str

class GitHubSettings(BaseModel):
    token: SecretStr
    username: str | None = None

class LogfireSettings(BaseModel):
    token: SecretStr | None = None

class LangSmithSettings(BaseModel):
    api_key: SecretStr | None = None
    endpoint: str = "https://api.smith.langchain.com"
    tracing_v2: bool = False

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Flat env vars loaded from .env
    groq_api_key: SecretStr = Field(validation_alias="GROQ_API_KEY")
    fall_groq_api_key: SecretStr | None = Field(default=None, validation_alias="FALL_GROQ_API_KEY")
    openrouter_api_key: SecretStr = Field(validation_alias="OPENROUTER_API_KEY")
    
    
    qdrant_api_key: SecretStr = Field(validation_alias="QDRANT_API_KEY")
    qdrant_end_point: str = Field(validation_alias="QDRANT_END_POINT")
    qdrant_cluster_id: str = Field(validation_alias="QDRANT_CLUSTER_ID")
    qdrant_collection_name: str = Field(default="personal_kb", validation_alias="QDRANT_COLLECTION_NAME")
    
    logfire_token: SecretStr | None = Field(default=None, validation_alias="LOGFIRE_TOKEN")
    
    langsmith_api_key: SecretStr | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langchain_endpoint: str = Field(default="https://api.smith.langchain.com", validation_alias="LANGCHAIN_ENDPOINT")
    langchain_tracing_v2: bool = Field(default=False, validation_alias="LANGCHAIN_TRACING_V2")
    
    github_token: SecretStr = Field(validation_alias="GITHUB_TOKEN")
    github_username: str | None = Field(default=None, validation_alias="GITHUB_USERNAME")
    
    hf_home: str = Field(default="/tmp/huggingface", validation_alias="HF_HOME")
    
    resume_path: Path = Field(default=PROJECT_ROOT / "assets" / "Muhammad_Umer_Khan_AI_Resume.pdf", validation_alias="RESUME_PATH")
    model_name: str = Field(default="openai/gpt-oss-120b", validation_alias="MODEL_NAME")
    guard_model_name: str = Field(default="llama-3.3-70b-versatile", validation_alias="GUARD_MODEL_NAME")
    embedding_model: str = Field(default="BAAI/bge-base-en-v1.5", validation_alias="EMBEDDING_MODEL")

    @field_validator("resume_path")
    @classmethod
    def validate_resume_path(cls, v: Path) -> Path:
        # Convert relative path to absolute relative to PROJECT_ROOT
        if not v.is_absolute():
            v = PROJECT_ROOT / v
        if not v.exists():
            raise FileNotFoundError(f"PDF resume file not found at: {v.resolve()}")
        return v

    @property
    def app(self) -> AppSettings:
        return AppSettings(
            resume_path=self.resume_path,
            model_name=self.model_name,
            guard_model_name=self.guard_model_name,
            embedding_model=self.embedding_model
        )

    @property
    def groq(self) -> GroqSettings:
        return GroqSettings(
            api_key=self.groq_api_key,
            fallback_api_key=self.fall_groq_api_key
        )


    @property
    def qdrant(self) -> QdrantSettings:
        return QdrantSettings(
            api_key=self.qdrant_api_key,
            endpoint=self.qdrant_end_point,
            cluster_id=self.qdrant_cluster_id,
            collection_name=self.qdrant_collection_name
        )

    @property
    def github(self) -> GitHubSettings:
        return GitHubSettings(
            token=self.github_token,
            username=self.github_username
        )

    @property
    def logfire(self) -> LogfireSettings:
        return LogfireSettings(
            token=self.logfire_token
        )

    @property
    def langsmith(self) -> LangSmithSettings:
        return LangSmithSettings(
            api_key=self.langsmith_api_key,
            endpoint=self.langchain_endpoint,
            tracing_v2=self.langchain_tracing_v2
        )

@lru_cache()
def get_settings() -> Settings:
    return Settings()
