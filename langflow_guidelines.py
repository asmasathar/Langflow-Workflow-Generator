LANGFLOW_GENERAL_RULES = """
## Langflow Workflow Rules

### CRITICAL: Connectivity is Derived from Content
To connect any two nodes you MUST first ensure a matching field exists on the target.
For Prompt nodes: add {variable} in the template text to create an input port named 'variable'.
Example: adding {memory} in the prompt template creates a 'memory' input port on the Prompt node.
This is the ABSOLUTE rule: no {variable} in template → no port → no edge. Always add the variable first.
Every curly-brace variable in a template MUST have exactly one incoming edge from a matching source.
Every outgoing edge MUST land on a real field/port that exists on the target node.

### ChatInput and ChatOutput are MANDATORY for conversational workflows
Any workflow involving a user message MUST have ChatInput and ChatOutput.
ChatInput is ALWAYS the entry point (source) — no incoming edges ever.
ChatOutput is ALWAYS the exit point (sink) — no outgoing edges ever.
Without these two the workflow cannot be tested in Langflow Playground.

### Port / Type Compatibility Table
Message output        → str input or Message input      (COMPATIBLE)
Data output           → Data input                      (COMPATIBLE)
LanguageModel output  → LanguageModel input only        (LLM agent/structured output)
text_output (Message) → ChatOutput.input_value          (COMPATIBLE — use this for LLM→Output)
model_output (LLM)    → Agent.llm or StructuredOutput.llm (COMPATIBLE)
Tool output           → Agent.tools                     (COMPATIBLE)
Memory.messages_text  → Prompt.{memory} variable port   (COMPATIBLE — requires {memory} in template)
Embeddings output     → VectorStore embedding input     (COMPATIBLE)

### Node ID Format
ComponentType-XXXXX   (5 random alphanumeric chars, no spaces)
Examples: ChatInput-jFwUm, PromptComponent-aB3kQ, LanguageModelComponent-rP0Oy

### API Key Convention
Never hardcode secrets. Use {ENV_VAR_NAME} as the value.
OpenAI key:       {OPENAI_API_KEY}
Anthropic key:    {ANTHROPIC_API_KEY}
OpenRouter key:   {OPENROUTER_API_KEY}

### OpenRouter LLM Setup
  provider = "OpenRouter"
  api_base = "https://openrouter.ai/api/v1"
  api_key  = "{OPENROUTER_API_KEY}"
  model    = "openai/gpt-4o"  or  "anthropic/claude-3-5-sonnet"  or  "meta-llama/llama-3-70b"

### Prompt Template Variables
  Wrap any variable in {curly_braces} — each creates a new input port on the Prompt node.
  Common variables:
    {memory}   → connect Memory.messages_text here
    {context}  → connect retrieved RAG results here
    {document} → connect file/document content here
    {input}    → connect ChatInput.message here (if not going directly to LLM)
  Use double braces {{}} for literal curly braces in the output text.

### Required Field Filling
  Before graph wiring, EVERY component must have its critical parameters set:
  Model components: provider, model_name, api_key
  Prompt components: template text with appropriate {variables}
  Memory components: session_id (use "user_message_id")
  VectorStore components: collection_name, embedding model
  Agent components: system_prompt, tools list
"""

LANGFLOW_EDGE_PATTERNS = """
## Proven Langflow Edge Patterns

### 1. Minimal Chatbot (2 nodes)
  ChatInput.message → LanguageModelComponent.input_value
  LanguageModelComponent.text_output → ChatOutput.input_value

### 2. Chatbot with System Prompt (3 nodes)
  ChatInput.message → LanguageModelComponent.input_value
  Prompt.prompt → LanguageModelComponent.system_message
  LanguageModelComponent.text_output → ChatOutput.input_value
  [Prompt template: "You are a helpful assistant."]

### 3. Memory Chatbot (4 nodes) — MOST IMPORTANT PATTERN
  ChatInput.message → LanguageModelComponent.input_value
  Memory.messages_text → Prompt.memory          ← requires {memory} in Prompt template
  Prompt.prompt → LanguageModelComponent.system_message
  LanguageModelComponent.text_output → ChatOutput.input_value
  [Prompt template: "You are a helpful assistant.\n\nConversation history:\n{memory}"]

### 4. Agent with Tools (variable nodes)
  ChatInput.message → Agent.input_value
  Prompt.prompt → Agent.system_prompt
  ToolComponent.component_as_tool → Agent.tools
  LanguageModelComponent.model_output → Agent.llm
  Agent.response → ChatOutput.input_value

### 5. RAG / Document Q&A (5–6 nodes) — CRITICAL: use ParseData between File and Prompt
  ChatInput.message → LanguageModelComponent.input_value
  File.data → ParseData.data
  ParseData.parsed_text → Prompt.context        ← requires {context} in Prompt template
  Prompt.prompt → LanguageModelComponent.system_message
  LanguageModelComponent.text_output → ChatOutput.input_value
  [Prompt template: "Answer using this context:\n{context}"]

  ALTERNATIVE (without ParseData, using File.message directly):
  File.message → Prompt.document                ← requires {document} in Prompt template
  ChatInput.message → LanguageModelComponent.input_value
  Prompt.prompt → LanguageModelComponent.system_message
  LanguageModelComponent.text_output → ChatOutput.input_value

### 6. VectorStore RAG (6 nodes)
  ChatInput.message → VectorStoreComponent.search_query
  VectorStoreComponent.search_results → ParseData.data
  ParseData.parsed_text → Prompt.context        ← requires {context} in Prompt template
  ChatInput.message → LanguageModelComponent.input_value
  Prompt.prompt → LanguageModelComponent.system_message
  LanguageModelComponent.text_output → ChatOutput.input_value
  [embedding: OpenAIEmbeddings.embeddings → VectorStoreComponent.embedding]

### 7. Structured Output (4 nodes)
  ChatInput.message → StructuredOutput.input_value
  LanguageModelComponent.model_output → StructuredOutput.llm
  StructuredOutput.dataframe_output → ParseData.input_data
  ParseData.parsed_text → ChatOutput.input_value

### 8. Sequential LLM Chain (3+ nodes)
  ChatInput.message → LanguageModelComponent.input_value
  Prompt.prompt → LanguageModelComponent.system_message
  LanguageModelComponent.text_output → LanguageModelComponent2.input_value
  LanguageModelComponent2.text_output → ChatOutput.input_value

### KEY RULE FOR MEMORY:
  When user needs memory/history:
    1. Add a Prompt component
    2. Put {memory} in the Prompt template text
    3. Connect Memory.messages_text → Prompt.memory
    4. Connect Prompt.prompt → LanguageModelComponent.system_message
    WITHOUT step 2, the memory port does NOT exist on Prompt and the edge will fail!

### KEY RULE FOR RAG / DOCUMENT CONTEXT:
  When user needs document/file/RAG context:
    1. Add a Prompt component
    2. Put {context} (or {document}) in the Prompt template text
    3. Connect ParseData.parsed_text → Prompt.context (or File.message → Prompt.document)
    4. Connect Prompt.prompt → LanguageModelComponent.system_message
    WITHOUT step 2, the context port does NOT exist on Prompt!

### KEY RULE FOR TOOL OUTPUTS:
  LanguageModel's model_output (type: LanguageModel) → Agent.llm ONLY
  LanguageModel's text_output  (type: Message)        → ChatOutput, LLM.input_value, Prompt vars
  Tool component's component_as_tool (type: Tool)     → Agent.tools ONLY
  NEVER route model_output to input_value, system_message, or any Prompt variable port.
"""

LANGFLOW_COMPONENT_FIELD_GUIDE = """
## Key Component Fields Reference

### ChatInput
  required: (none)
  optional: input_value (str), sender (str), session_id (str)
  outputs:  message (Message)

### ChatOutput
  required: input_value (Message or str)
  outputs:  message (Message)

### Prompt (PromptComponent)
  required: template (str with {variables})
  optional: {variable_name} input ports — created dynamically by {variables} in template
  outputs:  prompt (Message)
  IMPORTANT: Save the template FIRST so dynamic ports appear before connecting.

### LanguageModelComponent
  required: input_value (Message), provider or model_name
  optional: system_message (Message), api_key, temperature, max_tokens
  outputs:  text_output (Message), model_output (LanguageModel)

### OpenAIModel (or AzureOpenAIModel)
  required: model_name, api_key or {OPENAI_API_KEY}
  optional: temperature (float), max_tokens (int), system_message
  outputs:  text_output (Message), model_output (LanguageModel)

### Memory
  required: (none)
  optional: session_id (str — use "user_message_id"), memory_key (str)
  outputs:  messages_text (str), messages (list[Message])

### Agent
  required: input_value (Message), llm (LanguageModel)
  optional: system_prompt (Message), tools (list[Tool])
  outputs:  response (Message)

### VectorStoreComponent (Chroma, Pinecone, Weaviate, Qdrant, FAISS, PGVector, AstraDB)
  required: embedding (Embeddings), collection_name (str)
  optional: search_query (Message or str), search_type (str), number_of_results (int)
  outputs:  search_results (list[Data])
  NOTE: Connect ChatInput.message → VectorStoreComponent.search_query for retrieval.
        Connect OpenAIEmbeddings.embeddings → VectorStoreComponent.embedding.

### OpenAIEmbeddings (or any Embeddings component)
  required: model (str — e.g. "text-embedding-ada-002"), api_key (str)
  outputs:  embeddings (Embeddings)
  NOTE: Connect embeddings output → VectorStoreComponent.embedding.

### File (FileComponent)
  required: path (str)
  outputs:  message (Message), data (Data)

### TextInput
  required: input_value (str)
  outputs:  text (Message)

### ParseData (ParseDataComponent)
  required: data (Data or list[Data])
  optional: template (str)
  outputs:  parsed_text (Message)
  NOTE: Best practice for RAG: File.data → ParseData.data → ParseData.parsed_text → Prompt.{context}

### StructuredOutput
  required: input_value (Message), llm (LanguageModel), schema (str or dict)
  outputs:  dataframe_output (Data)

### APIRequest
  required: url (str), method (str — GET/POST)
  optional: headers (dict), body (dict)
  outputs:  data (Data)

### SplitText (or TextSplitter)
  required: data_input (Data or Message)
  optional: chunk_size (int — default 1000), chunk_overlap (int — default 200)
  outputs:  chunks (list[Data])

### Astra DB Vector Store
  required: collection_name (str), api_endpoint (str), token (str), embedding (Embeddings)
  outputs:  search_results (list[Data])
"""
