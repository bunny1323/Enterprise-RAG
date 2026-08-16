"""
Prompt Registry and Versioning Service.
Loads versioned prompt templates associated with assistant, model, and prompt version.
"""
from pathlib import Path
from pydantic import BaseModel


class PromptTemplate(BaseModel):
    version: str = "1.0"
    assistant_id: str = "default"
    model: str = "llava:13b"
    system_prompt: str
    user_prompt_template: str


class PromptRegistry:
    """
    Registry for loading and serving versioned prompts.
    """

    def __init__(self, prompt_dir: str = "./config/prompts") -> None:
        self._prompt_dir = Path(prompt_dir)

    def get_template(
        self,
        assistant_id: str = "default",
        version: str = "v1",
    ) -> PromptTemplate:
        """Get versioned prompt template."""
        return PromptTemplate(
            version=version,
            assistant_id=assistant_id,
            system_prompt=(
                "You are an enterprise AI assistant. "
                "Answer the user's question using ONLY the provided evidence. "
                "If evidence is insufficient, state that clearly."
            ),
            user_prompt_template="EVIDENCE:\n{evidence}\n\nQUESTION:\n{query}\n\nANSWER:",
        )
