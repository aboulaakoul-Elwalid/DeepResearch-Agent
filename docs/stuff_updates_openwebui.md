v0.6.40 - 2025-11-25
fixed
🗄️ A critical PostgreSQL user listing performance issue was resolved by removing a redundant count operation that caused severe database slowdowns and potential timeouts when viewing user lists in admin panels.
v0.6.39 - 2025-11-25
added
💬 A user list modal was added to channels, displaying all users with access and featuring search, sorting, and pagination capabilities. Commit
💬 Channel navigation now displays the total number of users with access to the channel. Commit
🔌 Tool servers and MCP connections now support function name filtering, allowing administrators to selectively enable or block specific functions using allow/block lists. Commit
⚡ A toggle to disable parallel embedding processing was added via "ENABLE_ASYNC_EMBEDDING", allowing sequential processing for rate-limited or resource-constrained local embedding setups. #19444
🔄 Various improvements were implemented across the frontend and backend to enhance performance, stability, and security.
🌐 Localization improvements were made for German (de-DE) and Portuguese (Brazil) translations.
fixed
📝 Inline citations now render correctly within markdown lists and nested elements instead of displaying as "undefined" values. #19452
👥 Group member selection now works correctly without randomly selecting other users or causing the user list to jump around. #19426
👥 Admin panel user list now displays the correct total user count and properly paginates 30 items per page after fixing database query issues with group member joins. #19429
🔍 Knowledge base reindexing now works correctly after resolving async execution chain issues by implementing threadpool workers for embedding operations. #19434
🖼️ OpenAI image generation now works correctly after fixing a connection adapter error caused by incorrect URL formatting. #19435
changed
🔧 BREAKING: Docling configuration has been consolidated from individual environment variables into a single "DOCLING_PARAMS" JSON configuration and now supports API key authentication via "DOCLING_API_KEY", requiring users to migrate existing Docling settings to the new format. #16841, #19427
🔧 The environment variable "REPLACE_IMAGE_URLS_IN_CHAT_RESPONSE" has been renamed to "ENABLE_CHAT_RESPONSE_BASE64_IMAGE_URL_CONVERSION" for naming consistency.
v0.6.38 - 2025-11-24
fixed
🔍 Hybrid search now works reliably after recent changes.
🛠️ Tool server saving now handles errors gracefully, preventing failed saves from impacting the UI.
🔐 SSO/OIDC code fixed to improve login reliability and better handle edge cases.
v0.6.37 - 2025-11-24
added
🔐 Granular sharing permissions are now available with two-tiered control separating group sharing from public sharing, allowing administrators to independently configure whether users can share workspace items with groups or make them publicly accessible, with separate permission toggles for models, knowledge bases, prompts, tools, and notes, configurable via "USER_PERMISSIONS_WORKSPACE_MODELS_ALLOW_SHARING", "USER_PERMISSIONS_WORKSPACE_MODELS_ALLOW_PUBLIC_SHARING", and corresponding environment variables for other workspace item types, while groups can now be configured to opt-out of sharing via the "Allow Group Sharing" setting. Commit, Commit
🔐 Password policy enforcement is now available with configurable validation rules, allowing administrators to require specific password complexity requirements via "ENABLE_PASSWORD_VALIDATION" and "PASSWORD_VALIDATION_REGEX_PATTERN" environment variables, with default pattern requiring minimum 8 characters including uppercase, lowercase, digit, and special character. #17794
🔐 Granular import and export permissions are now available for workspace items, introducing six separate permission toggles for models, prompts, and tools that are disabled by default for enhanced security. #19242
👥 Default group assignment is now available for new users, allowing administrators to automatically assign newly registered users to a specified group for streamlined access control to models, prompts, and tools, particularly useful for organizations with group-based model access policies. #19325, #17842
🔒 Password-based authentication can now be fully disabled via "ENABLE_PASSWORD_AUTH" environment variable, enforcing SSO-only authentication and preventing password login fallback when SSO is configured. #19113
🖼️ Large stream chunk handling was implemented to support models that generate images directly in their output responses, with configurable buffer size via "CHAT_STREAM_RESPONSE_CHUNK_MAX_BUFFER_SIZE" environment variable, resolving compatibility issues with models like Gemini 2.5 Flash Image. #18884, #17626
🖼️ Streaming response middleware now handles images in delta updates with automatic base64 conversion, enabling proper display of images from models using the "choices[0].delta.images.image_url" format such as Gemini 2.5 Flash Image Preview on OpenRouter. #19073, #19019
📈 Model list API performance was optimized by pre-fetching user group memberships and removing profile image URLs from response payloads, significantly reducing both database queries and payload size for instances with large model lists, with profile images now served dynamically via dedicated endpoints. #19097, #18950
⏩ Batch file processing performance was improved by reducing database queries by 67% while ensuring data consistency between vector and relational databases. #18953
🚀 Chat import performance was dramatically improved by replacing individual per-chat API requests with a bulk import endpoint, reducing import time by up to 95% for large chat collections and providing user feedback via toast notifications displaying the number of successfully imported chats. #17861
⚡ Socket event broadcasting performance was optimized by implementing user-specific rooms, significantly reducing server overhead particularly for users with multiple concurrent sessions. #18996
🗄️ Weaviate is now supported as a vector database option, providing an additional choice for RAG document storage alongside existing ChromaDB, Milvus, Qdrant, and OpenSearch integrations. #14747
🗄️ PostgreSQL pgvector now supports HNSW index types and large dimensional embeddings exceeding 2000 dimensions through automatic halfvec type selection, with configurable index methods via "PGVECTOR_INDEX_METHOD", "PGVECTOR_HNSW_M", "PGVECTOR_HNSW_EF_CONSTRUCTION", and "PGVECTOR_IVFFLAT_LISTS" environment variables. #19158, #16890
🔍 Azure AI Search is now supported as a web search provider, enabling integration with Azure's cognitive search services via "AZURE_AI_SEARCH_API_KEY", "AZURE_AI_SEARCH_ENDPOINT", and "AZURE_AI_SEARCH_INDEX_NAME" configuration. #19104
⚡ External embedding generation now processes API requests in parallel instead of sequential batches, reducing document processing time by 10-50x when using OpenAI, Azure OpenAI, or Ollama embedding providers, with large PDFs now processing in seconds instead of minutes. #19296
💨 Base64 image conversion is now available for markdown content in chat responses, automatically uploading embedded images exceeding 1KB and replacing them with file URLs to reduce payload size and resource consumption, configurable via "REPLACE_IMAGE_URLS_IN_CHAT_RESPONSE" environment variable. #19076
🎨 OpenAI image generation now supports additional API parameters including quality settings for GPT Image 1, configurable via "IMAGES_OPENAI_API_PARAMS" environment variable or through the admin interface, enabling cost-effective image generation with low, medium, or high quality options. #19228
🖼️ Image editing can now be independently enabled or disabled via admin settings, allowing administrators to control whether sequential image prompts trigger image editing or new image generation, configurable via "ENABLE_IMAGE_EDIT" environment variable. #19284
🔐 SSRF protection was implemented with a configurable URL blocklist that prevents access to cloud metadata endpoints and private networks, with default protections for AWS, Google Cloud, Azure, and Alibaba Cloud metadata services, customizable via "WEB_FETCH_FILTER_LIST" environment variable. #19201
⚡ Workspace models page now supports server-side pagination dramatically improving load times and usability for instances with large numbers of workspace models.
🔍 Hybrid search now indexes file metadata including filenames, titles, headings, sources, and snippets alongside document content, enabling keyword queries to surface documents where search terms appear only in metadata, configurable via "ENABLE_RAG_HYBRID_SEARCH_ENRICHED_TEXTS" environment variable. #19095
📂 Knowledge base upload page now supports folder drag-and-drop with recursive directory handling, enabling batch uploads of entire directory structures instead of requiring individual file selection. #19320
🤖 Model cloning is now available in admin settings, allowing administrators to quickly create workspace models based on existing base models through a "Clone" option in the model dropdown menu. #17937
🎨 UI scale adjustment is now available in interface settings, allowing users to increase the size of the entire interface from 1.0x to 1.5x for improved accessibility and readability, particularly beneficial for users with visual impairments. #19186
📌 Default pinned models can now be configured by administrators for all new users, mirroring the behavior of default models where admin-configured defaults apply only to users who haven't customized their pinned models, configurable via "DEFAULT_PINNED_MODELS" environment variable. #19273
🎙️ Text-to-Speech and Speech-to-Text services now receive user information headers when "ENABLE_FORWARD_USER_INFO_HEADERS" is enabled, allowing external TTS and STT providers to implement user-specific personalization, rate limiting, and usage tracking. #19323, #19312
🎙️ Voice mode now supports custom system prompts via "VOICE_MODE_PROMPT_TEMPLATE" configuration, allowing administrators to control response style and behavior for voice interactions. #18607
🔧 WebSocket and Redis configuration options are now available including debug logging controls, custom ping timeout and interval settings, and arbitrary Redis connection options via "WEBSOCKET_SERVER_LOGGING", "WEBSOCKET_SERVER_ENGINEIO_LOGGING", "WEBSOCKET_SERVER_PING_TIMEOUT", "WEBSOCKET_SERVER_PING_INTERVAL", and "WEBSOCKET_REDIS_OPTIONS" environment variables. #19091
🔧 MCP OAuth dynamic client registration now automatically detects and uses the appropriate token endpoint authentication method from server-supported options, enabling compatibility with OAuth servers that only support "client_secret_basic" instead of "client_secret_post". #19193
🔧 Custom headers can now be configured for remote MCP and OpenAPI tool server connections, enabling integration with services that require additional authentication headers. #18918
🔍 Perplexity Search now supports custom API endpoints via "PERPLEXITY_SEARCH_API_URL" configuration and automatically forwards user information headers to enable personalized search experiences. #19147
🔍 User information headers can now be optionally forwarded to external web search engines when "ENABLE_FORWARD_USER_INFO_HEADERS" is enabled. #19043
📊 Daily active user metric is now available for monitoring, tracking unique users active since midnight UTC via the "webui.users.active.today" Prometheus gauge. #19236, #19234
📊 Audit log file path is now configurable via "AUDIT_LOGS_FILE_PATH" environment variable, enabling storage in separate volumes or custom locations. #19173
🎨 Sidebar collapse states for model lists and group information are now persistent across page refreshes, remembering user preferences through browser-based storage. #19159
🎨 Background image display was enhanced with semi-transparent overlays for navbar and sidebar, creating a seamless and visually cohesive design across the entire interface. #19157
📋 Tables in chat messages now include a copy button that appears on hover, enabling quick copying of table content alongside the existing CSV export functionality. #19162
📝 Notes can now be created directly via the "/notes/new" URL endpoint with optional title and content query parameters, enabling faster note creation through bookmarks and shortcuts. #19195
🏷️ Tag suggestions are now context-aware, displaying only relevant tags when creating or editing models versus chat conversations, preventing confusion between model and chat tags. #19135
✍️ Prompt autocompletion is now available independently of the rich text input setting, improving accessibility to the feature. #19150
🔄 Various improvements were implemented across the frontend and backend to enhance performance, stability, and security.
🌐 Translations for Simplified Chinese, Traditional Chinese, Portuguese (Brazil), Catalan, Spanish (Spain), Finnish, Irish, Farsi, Swedish, Danish, German, Korean, and Thai were improved and expanded.
fixed
🤖 Model update functionality now works correctly, resolving a database parameter binding error that prevented saving changes to model configurations via the Save & Update button. #19335
🖼️ Multiple input images for image editing and generation are now correctly passed as an array using the "image[]" parameter syntax, enabling proper multi-image reference functionality with models like GPT Image 1. #19339
📱 PWA installations on iOS now properly refresh after server container restarts, resolving freezing issues by automatically unregistering service workers when version or deployment changes are detected. #19316
🗄️ S3 Vectors collection detection now correctly handles buckets with more than 2000 indexes by using direct index lookup instead of paginated list scanning, improving performance by approximately 8x and enabling RAG queries to work reliably at scale. #19238, #19233
📈 Feedback retrieval performance was optimized by eliminating N+1 query patterns through database joins, adding server-side pagination and sorting, significantly reducing database load for instances with large feedback datasets. #17976
🔍 Chat search now works correctly with PostgreSQL when chat data contains null bytes, with comprehensive sanitization preventing null bytes during data writes, cleaning existing data on read, and stripping null bytes during search queries to ensure reliable search functionality. #15616
🔍 Hybrid search with reranking now correctly handles attribute validation, preventing errors when collection results lack expected structure. #19025, #17046
🔎 Reranking functionality now works correctly after recent refactoring, resolving crashes caused by incorrect function argument handling. #19270
🤖 Azure OpenAI models now support the "reasoning_effort" parameter, enabling proper configuration of reasoning capabilities for models like GPT-5.1 which default to no reasoning without this setting. #19290
🤖 Models with very long IDs can now be deleted correctly, resolving URL length limitations that previously prevented management operations on such models. #18230
🤖 Model-level streaming settings now correctly apply to API requests, ensuring "Stream Chat Response" toggle properly controls the streaming parameter. #19154
🖼️ Image editing configuration now correctly preserves independent OpenAI API endpoints and keys, preventing them from being overwritten by image generation settings. #19003
🎨 Gemini image edit settings now display correctly in the admin panel, fixing an incorrect configuration key reference that prevented proper rendering of edit options. #19200
🖌️ Image generation settings menu now loads correctly, resolving validation errors with AUTOMATIC1111 API authentication parameters. #19187, #19246
📅 Date formatting in chat search and admin user chat search now correctly respects the "DEFAULT_LOCALE" environment variable, displaying dates according to the configured locale instead of always using MM/DD/YYYY format. #19305, #19020
📝 RAG template query placeholder escaping logic was corrected to prevent unintended replacements of context values when query placeholders appear in retrieved content. #19102, #19101
📄 RAG template prompt duplication was eliminated by removing redundant user query section from the default template. #19099, #19098
📋 MinerU local mode configuration no longer incorrectly requires an API key, allowing proper use of local content extraction without external API credentials. #19258
📊 Excel file uploads now work correctly with the addition of the missing msoffcrypto-tool dependency, resolving import errors introduced by the unstructured package upgrade. #19153
📑 Docling parameters now properly handle JSON serialization, preventing exceptions and ensuring configuration changes are saved correctly. #19072
🛠️ UserValves configuration now correctly isolates settings per tool, preventing configuration contamination when multiple tools with UserValves are used simultaneously. #19185, #15569
🔧 Tool selection prompt now correctly handles user messages without duplication, removing redundant query prefixes and improving prompt clarity. #19122, #19121
📝 Notes chat feature now correctly submits messages to the completions endpoint, resolving errors that prevented AI model interactions. #19079
📝 Note PDF downloads now sanitize HTML content using DOMPurify before rendering, preventing potential DOM-based XSS attacks from malicious content in notes. Commit
📁 Archived chats now have their folder associations automatically removed to prevent unintended deletion when their previous folder is deleted. #14578
🔐 ElevenLabs API key is now properly obfuscated in the admin settings page, preventing plain text exposure of sensitive credentials. #19262, #19260
🔧 MCP OAuth server metadata discovery now follows the correct specification order, ensuring proper authentication flow compliance. #19244
🔒 API key endpoint restrictions now properly enforce access controls for all endpoints including SCIM, preventing unintended access when "API_KEY_ALLOWED_ENDPOINTS" is configured. #19168
🔓 OAuth role claim parsing now supports both flat and nested claim structures, enabling compatibility with OAuth providers that deliver claims as direct properties on the user object rather than nested structures. #19286
🔑 OAuth MCP server verification now correctly extracts the access token value for authorization headers instead of sending the entire token dictionary. #19149, #19148
⚙️ OAuth dynamic client registration now correctly converts empty strings to None for optional fields, preventing validation failures in MCP package integration. #19144, #19129
🔐 OIDC authentication now correctly passes client credentials in access token requests, ensuring compatibility with providers that require these parameters per RFC 6749. #19132, #19131
🔗 OAuth client creation now respects configured token endpoint authentication methods instead of defaulting to basic authentication, preventing failures with servers that don't support basic auth. #19165
📋 Text copied from chat responses in Chrome now pastes without background formatting, improving readability when pasting into word processors. #19083
changed
🗄️ Group membership data storage was refactored from JSON arrays to a dedicated relational database table, significantly improving query performance and scalability for instances with large numbers of users and groups, while API responses now return member counts instead of full user ID arrays. #19239
📄 MinerU parameter handling was refactored to pass parameters directly to the API, improving flexibility and fixing VLM backend configuration. #19105, #18446
🔐 API key creation is now controlled by granular user and group permissions, with the "ENABLE_API_KEY" environment variable renamed to "ENABLE_API_KEYS" and disabled by default, requiring explicit configuration at both the global and user permission levels, while related environment variables "ENABLE_API_KEY_ENDPOINT_RESTRICTIONS" and "API_KEY_ALLOWED_ENDPOINTS" were renamed to "ENABLE_API_KEYS_ENDPOINT_RESTRICTIONS" and "API_KEYS_ALLOWED_ENDPOINTS" respectively. #18336
v0.6.36 - 2025-11-07
added
🔐 OAuth group parsing now supports configurable separators via the "OAUTH_GROUPS_SEPARATOR" environment variable, enabling proper handling of semicolon-separated group claims from providers like CILogon. #18987, #18979
fixed
🛠️ Tool calling functionality is restored by correcting asynchronous function handling in tool parameter updates. #18981
🖼️ The ComfyUI image edit workflow editor modal now opens correctly when clicking the Edit button. #18978
🔥 Firecrawl import errors are resolved by implementing lazy loading and using the correct class name. #18973
🔌 Socket.IO CORS warning is resolved by properly configuring CORS origins for Socket.IO connections. Commit