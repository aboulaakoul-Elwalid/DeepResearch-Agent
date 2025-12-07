"""
AutoReasonSearchWorkflow used by the interactive chat launcher.

This implementation wires up the LiteLLM client plus the MCP-backed tools
(`google_search`, `snippet_search`, optional `browse_webpage`) so interactive
chat can actually call tools when using API models like Gemini.
"""

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import typer
import yaml
from pydantic import Field

from dr_agent import GenerationConfig, LLMToolClient
from dr_agent.workflow import BaseWorkflow, BaseWorkflowConfiguration
from dr_agent.tool_interface.data_types import DocumentToolOutput
from dr_agent.tool_interface.tool_parsers import create_tool_parser
from dr_agent.tool_interface.mcp_tools import (
    RAG_MCP_PATH,
    ArabicBooksSearchTool,
    ExaSearchTool,
    ExaWebsetsSearchTool,
    JinaBrowseTool,
    SemanticScholarSnippetSearchTool,
    SerperBrowseTool,
    SerperSearchTool,
)

# Default prompt file for the CLI demo
DEFAULT_PROMPT_PATH = (
    Path(__file__).parent.parent
    / "dr_agent"
    / "shared_prompts"
    / "unified_tool_calling_cli.yaml"
)


class AutoReasonSearchWorkflow(BaseWorkflow):
    """
    Slim workflow wrapper for interactive chat.

    It currently routes everything through a single LLMToolClient without tools.
    That is sufficient to reproduce API/key issues (e.g., Gemini auth) and to
    keep the public API used by scripts intact.
    """

    class Configuration(BaseWorkflowConfiguration):
        # Core search agent settings
        search_agent_model_name: str = "gemini/gemini-2.0-flash"
        search_agent_tokenizer_name: Optional[str] = None
        search_agent_api_key: Optional[str] = None
        search_agent_base_url: Optional[str] = None
        search_agent_provider: Optional[str] = None
        search_agent_max_tokens: int = 8000
        search_agent_temperature: float = 1.0
        search_agent_max_tool_calls: int = 10
        search_agent_min_tool_calls: int = (
            0  # Minimum tool calls before allowing answer
        )
        default_dataset_name: Optional[str] = None
        prompt_path: Optional[str] = None  # Path to custom prompt YAML file

        # Optional browse agent fields
        use_browse_agent: bool = False
        browse_agent_model_name: Optional[str] = None
        browse_agent_tokenizer_name: Optional[str] = None
        browse_agent_api_key: Optional[str] = None
        browse_agent_base_url: Optional[str] = None
        browse_agent_max_tokens: int = 8000
        browse_agent_temperature: float = 0.3

        # Tool/search metadata (kept for config compatibility)
        tool_parser: Optional[str] = Field(default="v20250824")
        search_provider: str = "serper"  # serper|exa
        number_documents_to_search: int = 10
        search_timeout: int = 180
        browse_tool_name: Optional[str] = None
        browse_timeout: int = 180
        browse_max_pages_to_fetch: int = 10
        browse_context_char_length: int = 6000

        search_agent_fallback_enabled: bool = False
        search_agent_fallback_base_url: Optional[str] = None
        search_agent_fallback_model_name: Optional[str] = None
        search_agent_fallback_tokenizer_name: Optional[str] = None
        search_agent_fallback_api_key: Optional[str] = None
        search_agent_fallback_provider: Optional[str] = None

        # Exa search (optional)
        use_exa_search: bool = False
        exa_num_results: int = 5
        exa_include_domains: Optional[str] = None  # comma-separated
        exa_exclude_domains: Optional[str] = None  # comma-separated
        exa_use_autoprompt: bool = True
        exa_search_type: str = "neural"

        # Exa Websets (optional curated collections)
        use_exa_websets: bool = False
        exa_webset_id: Optional[str] = None
        exa_websets_num_results: int = 5
        exa_websets_include_domains: Optional[str] = None  # comma-separated
        exa_websets_exclude_domains: Optional[str] = None  # comma-separated

        # Arabic library MCP tool
        use_arabic_library: bool = False
        arabic_library_chroma_path: Optional[str] = Field(
            default_factory=lambda: os.getenv("ARABIC_BOOKS_CHROMA_PATH")
        )
        arabic_library_collection: str = Field(
            default_factory=lambda: os.getenv("ARABIC_BOOKS_COLLECTION", "arabic_books")
        )
        arabic_library_num_results: int = 5

        # MCP transport configuration
        # Default to in-process MCP to avoid socket binding issues
        mcp_transport: str = Field(default="FastMCPTransport")
        mcp_port: Optional[int] = None  # Used when transport=StreamableHttpTransport
        mcp_executable: str = Field(
            default_factory=lambda: os.getenv("MCP_EXECUTABLE", str(RAG_MCP_PATH))
        )

    _default_configuration_path: str = str(
        Path(__file__).with_suffix(".yaml").resolve()
    )

    def setup_components(self) -> None:
        """Initialise prompt config and the primary LLM client."""
        cfg = self.configuration

        # Use custom prompt path if specified, otherwise default
        if cfg.prompt_path:
            prompt_file = Path(cfg.prompt_path)
            if not prompt_file.is_absolute():
                prompt_file = (
                    Path(__file__).parent.parent
                    / "dr_agent"
                    / "shared_prompts"
                    / cfg.prompt_path
                )
        else:
            prompt_file = DEFAULT_PROMPT_PATH
        self.prompt_config = self._load_prompt(prompt_file)

        tool_parser = create_tool_parser(cfg.tool_parser or "v20250824")

        def _split_csv(value: Optional[str]) -> Optional[List[str]]:
            if not value:
                return None
            return [v.strip() for v in value.split(",") if v.strip()]

        # Build MCP-backed tools that match the prompt names
        tools = []

        # Primary search tool exposed to the model as "google_search"
        if cfg.search_provider.lower() == "exa":
            tools.append(
                ExaSearchTool(
                    tool_parser=tool_parser,
                    number_documents_to_search=cfg.number_documents_to_search,
                    timeout=cfg.search_timeout,
                    include_domains=_split_csv(cfg.exa_include_domains),
                    exclude_domains=_split_csv(cfg.exa_exclude_domains),
                    use_autoprompt=cfg.exa_use_autoprompt,
                    search_type=cfg.exa_search_type,
                    name="google_search",  # keep prompt compatibility
                    mcp_port=cfg.mcp_port,
                    transport_type=cfg.mcp_transport,
                    mcp_executable=cfg.mcp_executable,
                )
            )
        else:
            tools.append(
                SerperSearchTool(
                    tool_parser=tool_parser,
                    number_documents_to_search=cfg.number_documents_to_search,
                    timeout=cfg.search_timeout,
                    name="google_search",
                    mcp_port=cfg.mcp_port,
                    transport_type=cfg.mcp_transport,
                    mcp_executable=cfg.mcp_executable,
                )
            )

        tools.append(
            SemanticScholarSnippetSearchTool(
                tool_parser=tool_parser,
                number_documents_to_search=cfg.number_documents_to_search,
                timeout=cfg.search_timeout,
                name="snippet_search",
                mcp_port=cfg.mcp_port,
                transport_type=cfg.mcp_transport,
                mcp_executable=cfg.mcp_executable,
            )
        )

        if cfg.use_exa_search:
            tools.append(
                ExaSearchTool(
                    tool_parser=tool_parser,
                    number_documents_to_search=cfg.exa_num_results,
                    timeout=cfg.search_timeout,
                    include_domains=_split_csv(cfg.exa_include_domains),
                    exclude_domains=_split_csv(cfg.exa_exclude_domains),
                    use_autoprompt=cfg.exa_use_autoprompt,
                    search_type=cfg.exa_search_type,
                    name="exa_search",
                    mcp_port=cfg.mcp_port,
                    transport_type=cfg.mcp_transport,
                    mcp_executable=cfg.mcp_executable,
                )
            )

        if cfg.use_exa_websets:
            tools.append(
                ExaWebsetsSearchTool(
                    tool_parser=tool_parser,
                    number_documents_to_search=cfg.exa_websets_num_results,
                    timeout=cfg.search_timeout,
                    include_domains=_split_csv(cfg.exa_websets_include_domains),
                    exclude_domains=_split_csv(cfg.exa_websets_exclude_domains),
                    webset_id=cfg.exa_webset_id,
                    name="exa_websets_search",
                    mcp_port=cfg.mcp_port,
                    transport_type=cfg.mcp_transport,
                    mcp_executable=cfg.mcp_executable,
                )
            )

        if cfg.use_arabic_library:
            tools.append(
                ArabicBooksSearchTool(
                    tool_parser=tool_parser,
                    number_documents_to_search=cfg.arabic_library_num_results,
                    timeout=cfg.search_timeout,
                    chroma_path=cfg.arabic_library_chroma_path,
                    collection_name=cfg.arabic_library_collection,
                    name="search_arabic_books",
                    mcp_port=cfg.mcp_port,
                    transport_type=cfg.mcp_transport,
                    mcp_executable=cfg.mcp_executable,
                )
            )

        if cfg.use_browse_agent:
            browse_tool = (cfg.browse_tool_name or "jina").lower()
            if browse_tool == "serper":
                tools.append(
                    SerperBrowseTool(
                        tool_parser=tool_parser,
                        max_pages_to_fetch=cfg.browse_max_pages_to_fetch,
                        timeout=cfg.browse_timeout,
                        context_chars=cfg.browse_context_char_length,
                        name="browse_webpage",
                        mcp_port=cfg.mcp_port,
                        transport_type=cfg.mcp_transport,
                        mcp_executable=cfg.mcp_executable,
                    )
                )
            else:
                tools.append(
                    JinaBrowseTool(
                        tool_parser=tool_parser,
                        max_pages_to_fetch=cfg.browse_max_pages_to_fetch,
                        timeout=cfg.browse_timeout,
                        context_chars=cfg.browse_context_char_length,
                        name="browse_webpage",
                        mcp_port=cfg.mcp_port,
                        transport_type=cfg.mcp_transport,
                        mcp_executable=cfg.mcp_executable,
                    )
                )

        (
            resolved_base_url,
            resolved_model_name,
            resolved_tokenizer_name,
            resolved_api_key,
            resolved_provider,
        ) = self._select_search_agent_endpoint(cfg)
        generation_config = GenerationConfig(
            temperature=cfg.search_agent_temperature,
            max_tokens=cfg.search_agent_max_tokens,
        )

        self.search_client = LLMToolClient(
            model_name=resolved_model_name,
            tokenizer_name=resolved_tokenizer_name,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            custom_llm_provider=resolved_provider,
            tools=tools,
            generation_config=generation_config,
        )

    async def __call__(
        self,
        problem: str,
        dataset_name: Optional[str] = None,
        verbose: bool = False,
        step_callback: Optional[Any] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Execute the workflow for a single query.

        Returns a dict shaped like the previous implementation so callers such
        as `interactive_auto_search.py` continue to work.
        """
        cfg = self.configuration
        effective_dataset = dataset_name or cfg.default_dataset_name
        messages = self._build_messages(problem, effective_dataset)
        result = await self.search_client.generate_with_tools(
            messages,
            max_tool_calls=cfg.search_agent_max_tool_calls,
            min_tool_calls=cfg.search_agent_min_tool_calls,
            include_tool_results=True,
            verbose=verbose,
            temperature=cfg.search_agent_temperature,
            max_tokens=cfg.search_agent_max_tokens,
            on_step_callback=step_callback,
        )

        searched_links, browsed_links = self._collect_links(result.tool_calls)

        # Maintain a familiar result shape for downstream consumers.
        return {
            "generated_text": result.generated_text,
            "full_traces": result,
            "browsed_links": browsed_links,
            "searched_links": searched_links,
            "total_tool_calls": result.tool_call_count,
            "total_failed_tool_calls": len(
                [t for t in result.tool_calls if getattr(t, "error", None)]
            ),
        }

    # ---- Helpers ----
    def _load_prompt(self, prompt_path: Path) -> Dict[str, Any]:
        """Load prompt YAML; fall back to a simple system prompt if missing."""
        if prompt_path.exists():
            with prompt_path.open("r") as f:
                return yaml.safe_load(f) or {}
        return {
            "system_prompt": "You are a helpful research assistant. "
            "Think step by step and answer clearly.",
            "additional_instructions": {},
        }

    def _build_messages(
        self, problem: str, dataset_name: Optional[str]
    ) -> List[Dict[str, str]]:
        """Construct chat messages with optional dataset-specific guidance."""
        system_prompt = self.prompt_config.get("system_prompt", "").strip()
        additions = self.prompt_config.get("additional_instructions", {}) or {}

        if dataset_name:
            extra = additions.get(dataset_name)
            if not extra:
                # Common fallback keys for the provided prompt file.
                extra = additions.get("long_form")
            if extra:
                system_prompt = f"{system_prompt}\n\n{extra.strip()}"

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": problem})
        return messages

    def _collect_links(self, tool_calls: List[Any]) -> tuple[list[str], list[str]]:
        """Extract searched/browsed URLs from tool outputs."""
        searched_links: List[str] = []
        browsed_links: List[str] = []

        for call in tool_calls or []:
            if isinstance(call, DocumentToolOutput) and call.documents:
                for doc in call.documents:
                    if getattr(doc, "url", None):
                        if call.tool_name == "browse_webpage":
                            browsed_links.append(doc.url)
                        else:
                            searched_links.append(doc.url)
        return searched_links, browsed_links

    def _select_search_agent_endpoint(
        self, cfg: "AutoReasonSearchWorkflow.Configuration"
    ) -> Tuple[Optional[str], str, Optional[str], Optional[str], Optional[str]]:
        """Resolve the best base URL/model/provider or fall back if the primary endpoint is down."""
        base_url = (cfg.search_agent_base_url or "").strip() or None
        model_name = cfg.search_agent_model_name
        tokenizer_name = cfg.search_agent_tokenizer_name
        api_key = self._resolve_api_key(cfg.search_agent_api_key, model_name)
        provider = cfg.search_agent_provider or ("openai" if base_url else None)

        if (
            cfg.search_agent_fallback_enabled
            and base_url
            and not self._is_endpoint_reachable(base_url)
            and cfg.search_agent_fallback_model_name
        ):
            fallback_model = cfg.search_agent_fallback_model_name
            fallback_tokenizer = (
                cfg.search_agent_fallback_tokenizer_name
                if cfg.search_agent_fallback_tokenizer_name is not None
                else tokenizer_name
            )
            fallback_base_url = (
                cfg.search_agent_fallback_base_url or ""
            ).strip() or None
            fallback_api_key = self._resolve_api_key(
                cfg.search_agent_fallback_api_key, fallback_model
            )
            fallback_provider = cfg.search_agent_fallback_provider or (
                "openai" if fallback_base_url else None
            )

            print(
                f"⚠ Search endpoint {cfg.search_agent_base_url} unreachable, falling back to {fallback_model}"
            )
            if fallback_base_url:
                print(f"    Using fallback base URL: {fallback_base_url}")

            return (
                fallback_base_url,
                fallback_model,
                fallback_tokenizer,
                fallback_api_key,
                fallback_provider,
            )

        return base_url, model_name, tokenizer_name, api_key, provider

    def _is_endpoint_reachable(self, base_url: str) -> bool:
        """Quick health probe for OpenAI-compatible endpoints."""
        try:
            response = requests.head(base_url, timeout=3, allow_redirects=True)
            # Treat 5xx as unreachable; 2xx-4xx are acceptable (even 404 if root path not defined)
            return 200 <= response.status_code < 500
        except requests.RequestException:
            return False

    def _resolve_api_key(
        self, configured_key: Optional[str], model_name: str
    ) -> Optional[str]:
        """
        Prefer a real key over placeholders; fall back to common env vars so we
        don't pass dummy keys into LiteLLM.
        """
        placeholder_keys = {"", "dummy", "dummy-key", "your-key-here", None}
        if configured_key and configured_key not in placeholder_keys:
            return configured_key

        for env_var in self._suggest_env_vars(model_name):
            env_val = os.getenv(env_var)
            if env_val:
                return env_val
        return None  # Allow LiteLLM to use its own env discovery

    def _suggest_env_vars(self, model_name: str) -> List[str]:
        """Pick likely environment variables based on provider."""
        name = model_name.lower()
        if "groq" in name:
            return ["GROQ_API_KEY"]
        if "gemini" in name or "google" in name:
            return ["GOOGLE_API_KEY", "GOOGLE_AI_API_KEY", "GEMINI_API_KEY"]
        if "gpt" in name or "openai" in name or name.startswith("o1-"):
            return ["OPENAI_API_KEY"]
        return ["API_KEY"]


# ---- Simple CLI for direct calls ----
app = typer.Typer(help="AutoReasonSearch workflow utilities")


@app.command()
def chat(
    question: str = typer.Argument(..., help="User query to send to the workflow."),
    config: str = typer.Option(
        AutoReasonSearchWorkflow._default_configuration_path,
        "--config",
        "-c",
        help="Path to workflow YAML config.",
    ),
    dataset_name: Optional[str] = typer.Option(
        None, "--dataset-name", "-d", help="Optional dataset preset."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
):
    """Minimal CLI entrypoint to quickly exercise the workflow."""
    workflow = AutoReasonSearchWorkflow(configuration=config)
    output = asyncio.run(
        workflow(problem=question, dataset_name=dataset_name, verbose=verbose)
    )
    typer.echo(output["generated_text"])


if __name__ == "__main__":
    app()
