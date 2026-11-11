from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Config:
    model: str = "hyde/glm-5.3-flash"
    api_base: str = "https://inference.hyde.tech/v1"
    api_key: str = ""
    batch_size: int = 8
    max_archetypes: int = 30
    request_timeout: int = 90
    annotation_cutoff_minutes: int = 64
    random_seed: int = 7

    @classmethod
    def from_env(cls):
        return cls(api_key=os.getenv("HYDE_API_KEY", os.getenv("OPENAI_API_KEY", "")),
                   model=os.getenv("HYDE_MODEL", cls.model),
                   api_base=os.getenv("HYDE_API_BASE", cls.api_base),
                   annotation_cutoff_minutes=int(os.getenv("HYDE_ANNOTATION_CUTOFF_MINUTES", str(cls.annotation_cutoff_minutes))))
