"""Prompt configuration loader supporting YAML/JSON files.

Zero-dependency approach: uses standard library only, with optional PyYAML support.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PromptConfig:
    """Prompt configuration object.
    
    Attributes:
        template_id: Unique identifier for this template
        version: Semantic version string
        system: System prompt content
        user_template: User prompt template with {placeholders}
        variables: List of required variable names for formatting
    """
    template_id: str
    version: str
    system: str
    user_template: str
    variables: list[str]

    def render(self, **kwargs: Any) -> tuple[str, str]:
        """Render prompts with provided variables.
        
        Args:
            **kwargs: Variables to substitute in templates
            
        Returns:
            Tuple of (system_prompt, user_prompt)
            
        Raises:
            KeyError: If required variables are missing
            ValueError: If template format is invalid
        """
        missing = [v for v in self.variables if v not in kwargs]
        if missing:
            raise KeyError(f"Missing required variables: {missing}")
        
        try:
            user = self.user_template.format(**kwargs)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Template rendering failed: {exc}") from exc
        
        return self.system, user


class PromptRepository(Protocol):
    """Protocol for prompt storage backends."""
    
    def get(self, template_id: str, version: str | None = None) -> PromptConfig:
        """Get prompt configuration by ID and optional version."""
        ...
    
    def list_versions(self, template_id: str) -> list[str]:
        """List available versions for a template."""
        ...


class FileSystemPromptRepository:
    """File-based prompt repository supporting YAML and JSON formats.
    
    File structure:
        prompts/
            rubric_initializer.json
            rubric_proposer.json
            verifier.json
            judge.json
    
    Each file can contain either:
        - A single template object
        - A "templates" array with multiple versions
    
    Note: YAML support requires PyYAML to be installed:
        pip install pyyaml
    """
    
    def __init__(
        self, 
        base_path: str | Path = "prompts",
        default_extension: str = ".json",
    ) -> None:
        self.base_path = Path(base_path)
        self.default_extension = default_extension
        self._cache: dict[str, PromptConfig] = {}
        self._has_yaml = self._check_yaml()
        
        if not self.base_path.exists():
            logger.warning("Prompt directory not found: %s", self.base_path)
    
    def _check_yaml(self) -> bool:
        """Check if PyYAML is available."""
        try:
            import yaml
            return True
        except ImportError:
            return False
    
    def get(self, template_id: str, version: str | None = None) -> PromptConfig:
        """Get prompt configuration.
        
        Args:
            template_id: Template identifier (filename without extension)
            version: Specific version to load, or None for latest
            
        Returns:
            PromptConfig object
            
        Raises:
            FileNotFoundError: If template file not found
            ValueError: If version not found or config invalid
        """
        cache_key = f"{template_id}:{version or 'latest'}"
        
        if cache_key not in self._cache:
            config = self._load_template(template_id, version)
            self._cache[cache_key] = config
        
        return self._cache[cache_key]
    
    def list_versions(self, template_id: str) -> list[str]:
        """List all available versions for a template."""
        data = self._load_file(template_id)
        templates = data.get("templates", [data])
        return [t.get("version", "1.0.0") for t in templates]
    
    def clear_cache(self) -> None:
        """Clear internal cache to support hot-reloading."""
        self._cache.clear()
        logger.debug("Prompt cache cleared")
    
    def _load_file(self, template_id: str) -> dict[str, Any]:
        """Load and parse template file."""
        # Try JSON first (no dependencies needed)
        json_path = self.base_path / f"{template_id}.json"
        if json_path.exists():
            return json.loads(json_path.read_text(encoding="utf-8"))
        
        # Try YAML if PyYAML is available
        if self._has_yaml:
            for ext in [".yaml", ".yml"]:
                yaml_path = self.base_path / f"{template_id}{ext}"
                if yaml_path.exists():
                    import yaml
                    return yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        
        raise FileNotFoundError(
            f"No template file found for '{template_id}' in {self.base_path}. "
            f"Tried: .json{', .yaml, .yml' if self._has_yaml else ''}"
        )
    
    def _load_template(self, template_id: str, version: str | None = None) -> PromptConfig:
        """Load specific template version from file."""
        data = self._load_file(template_id)
        
        # Handle single template or multi-version
        templates = data.get("templates", [data])
        
        if version:
            template = next(
                (t for t in templates if t.get("version") == version),
                None
            )
            if template is None:
                available = [t.get("version", "unknown") for t in templates]
                raise ValueError(
                    f"Version '{version}' not found for template '{template_id}'. "
                    f"Available: {available}"
                )
        else:
            # Get latest version
            if not templates:
                raise ValueError(f"No templates found in file for '{template_id}'")
            template = templates[-1]
        
        # Validate required fields
        required = ["system", "user_template"]
        missing = [f for f in required if f not in template]
        if missing:
            raise ValueError(
                f"Template '{template_id}' missing required fields: {missing}"
            )
        
        return PromptConfig(
            template_id=template_id,
            version=template.get("version", "1.0.0"),
            system=template["system"],
            user_template=template["user_template"],
            variables=template.get("variables", []),
        )


class ConstantPromptRepository:
    """Repository that serves prompts from code constants.
    
    This is a fallback when no external configuration is needed.
    """
    
    def __init__(self, constants_module: Any | None = None) -> None:
        from . import constants
        self.constants = constants_module or constants
    
    def get(self, template_id: str, version: str | None = None) -> PromptConfig:
        """Get prompt from constants. Version is ignored."""
        registry = getattr(self.constants, "TEMPLATE_REGISTRY", {})
        
        if template_id not in registry:
            raise ValueError(f"Unknown template: {template_id}")
        
        meta = registry[template_id]
        return PromptConfig(
            template_id=template_id,
            version=version or "constant",
            system=meta["system"],
            user_template=meta["user_template"],
            variables=meta["variables"],
        )
    
    def list_versions(self, template_id: str) -> list[str]:
        """Constants only have one version."""
        return ["constant"]


def create_repository(
    source: str | Path | None = None,
    prefer_files: bool = True,
) -> PromptRepository:
    """Factory function to create appropriate repository.
    
    Args:
        source: Path to config directory, or None to use constants only
        prefer_files: If True and source exists, use FileSystemPromptRepository
        
    Returns:
        Configured PromptRepository instance
    """
    if prefer_files and source is not None:
        path = Path(source)
        if path.exists():
            logger.info("Using file-based prompt repository: %s", path)
            return FileSystemPromptRepository(path)
    
    logger.debug("Using constant prompt repository")
    return ConstantPromptRepository()
