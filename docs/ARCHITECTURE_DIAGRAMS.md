# DR-Tulu Agent Architecture Diagrams

This document contains comprehensive Mermaid diagrams illustrating the DR-Tulu deep research agent architecture.

## Table of Contents

1. [Overall Architecture Overview](#1-overall-architecture-overview)
2. [ReAct Agent Loop Flow](#2-react-agent-loop-flow)
3. [Tool Registration & Execution Flow](#3-tool-registration--execution-flow)
4. [MCP Communication Pattern](#4-mcp-communication-pattern)
5. [Configuration Loading Process](#5-configuration-loading-process)
6. [Tool Parser Hierarchy](#6-tool-parser-hierarchy)
7. [Component Class Diagram](#7-component-class-diagram)
8. [Data Flow Sequence](#8-data-flow-sequence)

---

## 1. Overall Architecture Overview

This diagram shows the high-level architecture of the DR-Tulu agent system.

```mermaid
flowchart TB
    subgraph User["👤 User Interface"]
        CLI["CLI / Scripts"]
        Gateway["OpenAI Gateway<br/>dr_tulu_cli_gateway.py"]
    end

    subgraph Workflow["🔄 Workflow Layer"]
        WF["AutoReasonSearchWorkflow<br/>workflows/auto_search_sft.py"]
        WFConfig["YAML Config<br/>workflows/*.yaml"]
    end

    subgraph Agent["🤖 Agent Layer"]
        Client["LLMToolClient<br/>dr_agent/client.py"]
        AgentBase["AgentInterface<br/>dr_agent/agent_interface.py"]
    end

    subgraph LLM["🧠 LLM Providers"]
        LiteLLM["LiteLLM Adapter"]
        OpenAI["OpenAI API"]
        Gemini["Gemini API"]
        Local["Local Models<br/>(Parallax/vLLM)"]
    end

    subgraph Tools["🛠️ Tool Layer"]
        ToolBase["BaseTool<br/>tool_interface/base.py"]
        MCPSearch["MCPSearchTool"]
        MCPBrowse["MCPBrowseTool"]
        ChainedTool["ChainedTool"]
    end

    subgraph MCP["📡 MCP Backend"]
        MCPServer["FastMCP Server<br/>mcp_backend/main.py"]
        subgraph APIs["External APIs"]
            Serper["Serper API"]
            Exa["Exa API"]
            Scholar["Semantic Scholar"]
            Jina["Jina Reader"]
            Crawl4AI["Crawl4AI"]
        end
    end

    subgraph Parsers["📝 Tool Parsers"]
        Parser["ToolCallParser<br/>tool_parsers.py"]
        XMLParsers["v20250824 / Legacy /<br/>Unified / Qwen"]
    end

    CLI --> Gateway
    Gateway --> WF
    WF --> WFConfig
    WF --> Client
    Client --> AgentBase
    Client --> LiteLLM
    LiteLLM --> OpenAI
    LiteLLM --> Gemini
    LiteLLM --> Local
    Client --> Parser
    Parser --> XMLParsers
    Client --> ToolBase
    ToolBase --> MCPSearch
    ToolBase --> MCPBrowse
    ToolBase --> ChainedTool
    MCPSearch --> MCPServer
    MCPBrowse --> MCPServer
    MCPServer --> APIs
```

---

## 2. ReAct Agent Loop Flow

This diagram illustrates the core ReAct (Reasoning + Acting) loop implemented in `LLMToolClient.generate_with_tools()`.

```mermaid
flowchart TD
    Start([Start Query]) --> Init["Initialize<br/>• message_history = []<br/>• iteration = 0<br/>• max_iterations = 20"]
    
    Init --> BuildPrompt["Build System Prompt<br/>+ Tool Descriptions"]
    
    BuildPrompt --> LLMCall["🧠 LLM Generate<br/>get_litellm_response()"]
    
    LLMCall --> ParseResponse{"Parse Response<br/>for Tool Calls"}
    
    ParseResponse -->|"Tool Call Found<br/>(parser mode)"| ExtractTool["Extract Tool Call<br/>via ToolCallParser"]
    ParseResponse -->|"Tool Call Found<br/>(native mode)"| NativeExtract["Extract from<br/>tool_calls field"]
    ParseResponse -->|"No Tool Call"| CheckStop{"Check Stop<br/>Conditions"}
    
    ExtractTool --> ExecuteTool["🛠️ Execute Tool<br/>tool.execute()"]
    NativeExtract --> ExecuteTool
    
    ExecuteTool --> FormatResult["Format Tool Result<br/>as Message"]
    
    FormatResult --> AppendHistory["Append to<br/>message_history"]
    
    AppendHistory --> IncrementIter["iteration += 1"]
    
    IncrementIter --> CheckIter{"iteration ≥<br/>max_iterations?"}
    
    CheckIter -->|Yes| ForceStop["Force Stop<br/>Return Last Response"]
    CheckIter -->|No| LLMCall
    
    CheckStop -->|"<answer> tag found"| ExtractAnswer["Extract Answer<br/>from tags"]
    CheckStop -->|"Conclusion marker found"| ExtractAnswer
    CheckStop -->|"Max tokens reached"| ForceStop
    CheckStop -->|"Continue"| LLMCall
    
    ExtractAnswer --> End([Return Final Answer])
    ForceStop --> End

    style LLMCall fill:#e1f5fe
    style ExecuteTool fill:#fff3e0
    style ExtractAnswer fill:#e8f5e9
```

---

## 3. Tool Registration & Execution Flow

This diagram shows how tools are registered and executed within the agent system.

```mermaid
flowchart TD
    subgraph Registration["Tool Registration Phase"]
        WFInit["Workflow.__init__()"] --> LoadConfig["Load YAML Config"]
        LoadConfig --> CreateTools["Create Tool Instances"]
        
        CreateTools --> MCPSearch["MCPSearchTool<br/>• serper_web_search<br/>• exa_search<br/>• semantic_scholar_search"]
        CreateTools --> MCPBrowse["MCPBrowseTool<br/>• jina_read_url<br/>• crawl4ai_scrape"]
        CreateTools --> Chained["ChainedTool<br/>(Optional)"]
        
        MCPSearch --> RegisterClient["Register with<br/>LLMToolClient"]
        MCPBrowse --> RegisterClient
        Chained --> RegisterClient
        
        RegisterClient --> BuildSchema["Build Tool Schema<br/>for LLM"]
    end

    subgraph Execution["Tool Execution Phase"]
        ToolCall["Tool Call Detected"] --> ParseArgs["Parse Arguments<br/>from XML/JSON"]
        
        ParseArgs --> Lookup["Lookup Tool<br/>by Name"]
        
        Lookup --> ValidateArgs["Validate Arguments<br/>against Schema"]
        
        ValidateArgs -->|Valid| Execute["tool.execute(**args)"]
        ValidateArgs -->|Invalid| Error["Return Error Message"]
        
        Execute --> MCPTransport{"Transport Type?"}
        
        MCPTransport -->|HTTP| HTTPCall["HTTP Request to<br/>MCP Server"]
        MCPTransport -->|In-Memory| DirectCall["Direct FastMCP<br/>Function Call"]
        
        HTTPCall --> ExternalAPI["External API<br/>(Serper/Exa/etc.)"]
        DirectCall --> ExternalAPI
        
        ExternalAPI --> FormatResponse["Format Response<br/>as Tool Result"]
        
        FormatResponse --> ReturnToAgent["Return to Agent<br/>for Next Iteration"]
    end

    BuildSchema -.->|"At Runtime"| ToolCall

    style Registration fill:#e3f2fd
    style Execution fill:#fff8e1
```

---

## 4. MCP Communication Pattern

This diagram illustrates the Model Context Protocol (MCP) communication architecture.

```mermaid
flowchart LR
    subgraph Agent["Agent Process"]
        Client["LLMToolClient"]
        MCPTool["MCPSearchTool /<br/>MCPBrowseTool"]
    end

    subgraph Transport["Transport Layer"]
        HTTP["StreamableHttpTransport<br/>(HTTP/SSE)"]
        InMem["FastMCPTransport<br/>(In-Memory)"]
    end

    subgraph MCPServer["MCP Server Process"]
        FastMCP["FastMCP Server<br/>mcp_backend/main.py"]
        
        subgraph ToolHandlers["Tool Handlers"]
            SerperHandler["@mcp.tool()<br/>serper_web_search"]
            ExaHandler["@mcp.tool()<br/>exa_search"]
            ScholarHandler["@mcp.tool()<br/>semantic_scholar_search"]
            JinaHandler["@mcp.tool()<br/>jina_read_url"]
            CrawlHandler["@mcp.tool()<br/>crawl4ai_scrape"]
        end
    end

    subgraph External["External Services"]
        SerperAPI["Serper API<br/>google.serper.dev"]
        ExaAPI["Exa API<br/>api.exa.ai"]
        ScholarAPI["Semantic Scholar<br/>api.semanticscholar.org"]
        JinaAPI["Jina Reader<br/>r.jina.ai"]
        Crawl4AIAPI["Crawl4AI<br/>(Local/Remote)"]
    end

    Client --> MCPTool
    MCPTool --> HTTP
    MCPTool --> InMem
    HTTP -->|"JSON-RPC over HTTP"| FastMCP
    InMem -->|"Direct Python Call"| FastMCP
    
    FastMCP --> ToolHandlers
    SerperHandler --> SerperAPI
    ExaHandler --> ExaAPI
    ScholarHandler --> ScholarAPI
    JinaHandler --> JinaAPI
    CrawlHandler --> Crawl4AIAPI

    style Agent fill:#e8eaf6
    style Transport fill:#fce4ec
    style MCPServer fill:#e0f2f1
    style External fill:#fff3e0
```

---

## 5. Configuration Loading Process

This diagram shows how configuration flows from YAML files through to runtime objects.

```mermaid
flowchart TD
    subgraph YAMLFiles["YAML Configuration Files"]
        Basic["auto_search_basic.yaml"]
        Deep["auto_search_deep.yaml"]
        Parallax["auto_search_parallax.yaml"]
        Custom["Custom *.yaml"]
    end

    subgraph Loading["Configuration Loading"]
        CLI["CLI Arguments<br/>--config path"] --> ResolvePath["Resolve Config Path"]
        ResolvePath --> OmegaConf["OmegaConf.load()"]
        
        Basic --> OmegaConf
        Deep --> OmegaConf
        Parallax --> OmegaConf
        Custom --> OmegaConf
        
        OmegaConf --> Merge["Merge with CLI Overrides"]
        Merge --> DictConfig["DictConfig Object"]
    end

    subgraph ConfigSections["Configuration Sections"]
        DictConfig --> LLMConfig["llm_config:<br/>• model<br/>• api_base<br/>• temperature"]
        DictConfig --> ToolConfig["tool_config:<br/>• mcp_endpoint<br/>• search_tools<br/>• browse_tools"]
        DictConfig --> AgentConfig["agent_config:<br/>• max_iterations<br/>• tool_calling_mode<br/>• tool_parser"]
        DictConfig --> PromptConfig["prompt_config:<br/>• system_prompt_file<br/>• tool_prompt_file"]
    end

    subgraph Runtime["Runtime Objects"]
        LLMConfig --> ClientInit["LLMToolClient()<br/>initialization"]
        ToolConfig --> ToolInit["Tool Registration"]
        AgentConfig --> LoopConfig["ReAct Loop<br/>Parameters"]
        PromptConfig --> PromptLoad["Load Prompt<br/>Templates"]
        
        ClientInit --> Workflow["Configured Workflow"]
        ToolInit --> Workflow
        LoopConfig --> Workflow
        PromptLoad --> Workflow
    end

    style YAMLFiles fill:#e1f5fe
    style Loading fill:#f3e5f5
    style ConfigSections fill:#e8f5e9
    style Runtime fill:#fff8e1
```

---

## 6. Tool Parser Hierarchy

This diagram shows the tool parser class hierarchy and parsing strategies.

```mermaid
flowchart TD
    subgraph ParserSelection["Parser Selection"]
        Config["tool_parser config"] --> Switch{"Parser Type?"}
        
        Switch -->|"v20250824"| V2024["V20250824Parser"]
        Switch -->|"legacy"| Legacy["LegacyToolCallParser"]
        Switch -->|"unified"| Unified["UnifiedToolCallParser"]
        Switch -->|"qwen"| Qwen["QwenToolCallParser"]
    end

    subgraph ParserDetails["Parser Format Details"]
        V2024 --> V2024Format["Format:<br/>&lt;call_tool name='tool_name'&gt;<br/>  content/args<br/>&lt;/call_tool&gt;"]
        
        Legacy --> LegacyFormat["Format:<br/>&lt;tool_call&gt;<br/>  &lt;name&gt;tool_name&lt;/name&gt;<br/>  &lt;arguments&gt;...&lt;/arguments&gt;<br/>&lt;/tool_call&gt;"]
        
        Unified --> UnifiedFormat["Format:<br/>&lt;function=tool_name&gt;<br/>  {json_args}<br/>&lt;/function&gt;"]
        
        Qwen --> QwenFormat["Format:<br/>&lt;tool_call&gt;<br/>  {&quot;name&quot;: &quot;...&quot;, &quot;arguments&quot;: ...}<br/>&lt;/tool_call&gt;"]
    end

    subgraph ParseProcess["Parsing Process"]
        LLMOutput["LLM Output Text"] --> Detect["Detect Tool Call<br/>Pattern"]
        
        Detect --> Extract["Extract:<br/>• Tool Name<br/>• Arguments"]
        
        Extract --> Validate["Validate against<br/>Tool Schema"]
        
        Validate -->|Valid| ToolCallObj["ToolCall Object<br/>• name: str<br/>• arguments: dict"]
        Validate -->|Invalid| ParseError["ParseError<br/>+ Error Message"]
    end

    subgraph StopDetection["Stop Condition Detection"]
        AnswerTag["&lt;answer&gt;...&lt;/answer&gt;"]
        Conclusion["## Conclusion / Final Answer"]
        MaxIter["max_iterations reached"]
        
        AnswerTag --> Stop["Stop Loop"]
        Conclusion --> Stop
        MaxIter --> Stop
    end

    V2024Format --> LLMOutput
    LegacyFormat --> LLMOutput
    UnifiedFormat --> LLMOutput
    QwenFormat --> LLMOutput

    style ParserSelection fill:#e3f2fd
    style ParserDetails fill:#fce4ec
    style ParseProcess fill:#e8f5e9
    style StopDetection fill:#fff3e0
```

---

## 7. Component Class Diagram

This diagram shows the main classes and their relationships.

```mermaid
classDiagram
    class Workflow {
        <<abstract>>
        +config: DictConfig
        +load_config(path) DictConfig
        +run(query) str
    }

    class AutoReasonSearchWorkflow {
        +client: LLMToolClient
        +tools: List[BaseTool]
        +run(query) str
        +setup_tools()
    }

    class LLMToolClient {
        +model: str
        +api_base: str
        +tools: List[BaseTool]
        +tool_parser: ToolCallParser
        +max_iterations: int
        +generate_with_tools(messages) str
        +get_litellm_response(messages) str
        +execute_tool(tool_call) str
    }

    class AgentInterface {
        <<abstract>>
        +generate(prompt) str
        +chat(messages) str
    }

    class BaseTool {
        <<abstract>>
        +name: str
        +description: str
        +parameters: dict
        +execute(**kwargs) str
        +get_schema() dict
    }

    class MCPSearchTool {
        +transport: MCPTransport
        +tool_name: str
        +execute(**kwargs) str
    }

    class MCPBrowseTool {
        +transport: MCPTransport
        +tool_name: str
        +execute(**kwargs) str
    }

    class ChainedTool {
        +tools: List[BaseTool]
        +execute(**kwargs) str
    }

    class ToolCallParser {
        <<abstract>>
        +parse(text) ToolCall
        +detect_tool_call(text) bool
    }

    class V20250824Parser {
        +parse(text) ToolCall
        +pattern: regex
    }

    class MCPTransport {
        <<abstract>>
        +call_tool(name, args) str
    }

    class StreamableHttpTransport {
        +endpoint: str
        +call_tool(name, args) str
    }

    class FastMCPTransport {
        +mcp_server: FastMCP
        +call_tool(name, args) str
    }

    Workflow <|-- AutoReasonSearchWorkflow
    AutoReasonSearchWorkflow *-- LLMToolClient
    AutoReasonSearchWorkflow *-- BaseTool
    LLMToolClient --> AgentInterface
    LLMToolClient *-- ToolCallParser
    BaseTool <|-- MCPSearchTool
    BaseTool <|-- MCPBrowseTool
    BaseTool <|-- ChainedTool
    ToolCallParser <|-- V20250824Parser
    MCPSearchTool *-- MCPTransport
    MCPBrowseTool *-- MCPTransport
    MCPTransport <|-- StreamableHttpTransport
    MCPTransport <|-- FastMCPTransport
```

---

## 8. Data Flow Sequence

This sequence diagram shows the complete data flow for a research query.

```mermaid
sequenceDiagram
    participant User
    participant Gateway as OpenAI Gateway
    participant Workflow as AutoReasonSearchWorkflow
    participant Client as LLMToolClient
    participant LLM as LLM Provider
    participant Parser as ToolCallParser
    participant Tool as MCPSearchTool
    participant MCP as MCP Server
    participant API as External API

    User->>Gateway: POST /chat/completions
    Gateway->>Workflow: run(query)
    Workflow->>Client: generate_with_tools(messages)
    
    loop ReAct Loop (max 20 iterations)
        Client->>LLM: generate(messages + tool_schema)
        LLM-->>Client: response with tool call
        
        Client->>Parser: parse(response)
        Parser-->>Client: ToolCall(name, args)
        
        Client->>Tool: execute(**args)
        Tool->>MCP: call_tool(name, args)
        MCP->>API: HTTP request
        API-->>MCP: API response
        MCP-->>Tool: formatted result
        Tool-->>Client: tool result
        
        Client->>Client: append to message_history
        
        alt Answer tag detected
            Client-->>Workflow: final answer
        else Max iterations reached
            Client-->>Workflow: last response
        end
    end
    
    Workflow-->>Gateway: answer
    Gateway-->>User: OpenAI-format response
```

---

## Key Architecture Insights

### ReAct Loop Implementation
- Located in `LLMToolClient.generate_with_tools()` at `dr_agent/client.py:539-1521`
- Maximum 20 iterations by default
- Stops on `<answer>` tags, conclusion markers, or iteration limit

### Tool Calling Modes
1. **Parser Mode**: XML pattern detection using configurable parsers
2. **Native Mode**: OpenAI function calling format

### MCP Transport Options
1. **HTTP Transport**: `StreamableHttpTransport` for remote MCP servers
2. **In-Memory Transport**: `FastMCPTransport` for co-located servers

### Default Parser
- `v20250824` format: `<call_tool name="tool_name">arguments</call_tool>`

### Stop Conditions
1. `<answer>` tag detected in response
2. Conclusion markers (e.g., "## Conclusion", "Final Answer")
3. Maximum tool calls reached
4. Token limit exceeded
5. Iteration limit (20) reached

---

## File References

| Component | File Path |
|-----------|-----------|
| Main Client | `dr_agent/client.py` |
| Workflow Base | `dr_agent/workflow.py` |
| Agent Interface | `dr_agent/agent_interface.py` |
| Tool Base | `dr_agent/tool_interface/base.py` |
| MCP Tools | `dr_agent/tool_interface/mcp_tools.py` |
| Tool Parsers | `dr_agent/tool_interface/tool_parsers.py` |
| MCP Backend | `dr_agent/mcp_backend/main.py` |
| System Prompts | `dr_agent/shared_prompts/unified_tool_calling.py` |
| Auto Search Workflow | `workflows/auto_search_sft.py` |
| Config Examples | `workflows/*.yaml` |
