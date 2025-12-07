/* eslint-disable react-refresh/only-export-components */
import {
  createContext,
  useContext,
  useMemo,
  useState,
  type Dispatch,
  type FC,
  type PropsWithChildren,
  type SetStateAction,
} from 'react';
// API base URL is resolved dynamically; see connect() below.
import { useConst, useRefCallback } from '../hooks';
import { useCluster } from './cluster';
import { parseGenerationGpt, parseGenerationQwen } from './chat-helper';
import { getApiBaseUrl } from './config';

// Helpers
const parseSourcesFromPayload = (payload: unknown): Source[] => {
  const normalizeArray = (arr: any[]): Source[] => {
    return arr
      .map((item) => {
        if (!item) return undefined;
        if (typeof item === 'string') {
          return { id: crypto.randomUUID(), title: item };
        }
        if (typeof item === 'object') {
          const { id, title, url, link, snippet, summary, score, tool_call_id } = item as any;
          return {
            id: id || crypto.randomUUID(),
            title: title || item.name,
            url: url || link,
            snippet: snippet || summary,
            score,
            toolCallId: tool_call_id,
          };
        }
        return undefined;
      })
      .filter(Boolean) as Source[];
  };

  try {
    if (typeof payload === 'string') {
      const trimmed = payload.trim();
      if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
        const parsed = JSON.parse(trimmed);
        return parseSourcesFromPayload(parsed);
      }
      return [];
    }
    if (Array.isArray(payload)) {
      return normalizeArray(payload);
    }
    if (payload && typeof payload === 'object') {
      const obj = payload as Record<string, any>;
      if (Array.isArray(obj.sources)) {
        return normalizeArray(obj.sources);
      }
      if (Array.isArray(obj.documents)) {
        return normalizeArray(obj.documents);
      }
    }
  } catch (error) {
    // swallow parsing errors silently
  }
  return [];
};

const debugLog = async (...args: any[]) => {
  if (import.meta.env.DEV) {
    console.log('%c chat.tsx ', 'color: white; background: orange;', ...args);
  }
};

const mergeSources = (prev: readonly Source[] = [], incoming: readonly Source[]): Source[] => {
  if (!incoming.length) return prev as Source[];
  const seen = new Set<string>();
  const result: Source[] = [];
  const push = (source: Source) => {
    const key = source.url || source.id || source.title || crypto.randomUUID();
    if (seen.has(key)) return;
    seen.add(key);
    result.push(source);
  };
  prev.forEach(push);
  incoming.forEach(push);
  return result;
};

const extractTextDelta = (deltaContent: any): string => {
  if (typeof deltaContent === 'string') return deltaContent;
  if (Array.isArray(deltaContent)) {
    return deltaContent
      .map((part) => {
        if (!part) return '';
        if (typeof part === 'string') return part;
        if (typeof part === 'object' && part.type === 'text' && typeof part.text === 'string') {
          return part.text;
        }
        return '';
      })
      .join('');
  }
  return '';
};

export type ChatMessageRole = 'user' | 'assistant';

export type ChatMessageStatus = 'waiting' | 'thinking' | 'generating' | 'done' | 'error';

export interface ToolCall {
  readonly id: string;
  readonly name: string;
  readonly arguments: string;
  readonly status?: 'pending' | 'called' | 'responded';
}

export interface ToolOutput {
  readonly toolCallId: string;
  readonly content: string;
  readonly summary?: string;
  readonly sources?: readonly Source[];
}

export interface Source {
  readonly id: string;
  readonly title?: string;
  readonly url?: string;
  readonly snippet?: string;
  readonly score?: number;
  readonly toolCallId?: string;
}

export interface ChatMessage {
  readonly id: string;
  readonly role: ChatMessageRole;
  readonly status: ChatMessageStatus;

  /**
   * The content from user input or assistant generating.
   */
  readonly content: string;

  /**
   * Reasoning / thinking content streamed separately from the final answer.
   */
  readonly reasoning?: string;

  /**
   * The raw content from model response.
   */
  readonly raw?: string;

  /**
   * The thinking content in assistant generating.
   */
  readonly thinking?: string;

  /**
   * Tool calls triggered by the assistant.
   */
  readonly toolCalls?: readonly ToolCall[];

  /**
   * Outputs returned by tools.
   */
  readonly toolOutputs?: readonly ToolOutput[];

  /**
   * References / citations emitted by the model or tools.
   */
  readonly sources?: readonly Source[];

  /**
   * Raw trace log (e.g., “Search & Reasoning Trace”) for debug/inspection.
   */
  readonly traceRaw?: string;
  readonly createdAt: number;
}

export type ChatStatus = 'closed' | 'opened' | 'generating' | 'error';

export interface ChatStates {
  readonly input: string;
  readonly status: ChatStatus;
  readonly messages: readonly ChatMessage[];
}

export interface ChatActions {
  readonly setInput: Dispatch<SetStateAction<string>>;
  readonly generate: (message?: ChatMessage) => void;
  readonly stop: () => void;
  readonly clear: () => void;
}

export const ChatProvider: FC<PropsWithChildren> = ({ children }) => {
  const [
    {
      clusterInfo: { status: clusterStatus, modelName },
    },
  ] = useCluster();

  const [input, setInput] = useState<string>('');

  const [status, _setStatus] = useState<ChatStatus>('closed');
  const setStatus = useRefCallback<typeof _setStatus>((value) => {
    _setStatus((prev) => {
      const next = typeof value === 'function' ? value(prev) : value;
      if (next !== prev) {
        debugLog('setStatus', 'status', next);
      }
      return next;
    });
  });

  const [messages, setMessages] = useState<readonly ChatMessage[]>([]);

  const sse = useConst(() =>
    createSSE({
      onOpen: () => {
        debugLog('SSE OPEN');
        setStatus('opened');
      },
      onClose: () => {
        debugLog('SSE CLOSE');
        setMessages((prev) => {
          const lastMessage = prev[prev.length - 1];
          if (!lastMessage || lastMessage.role !== 'assistant') return prev;
          return [...prev.slice(0, -1), { ...lastMessage, status: 'done' as const }];
        });
        setStatus('closed');
      },
      onError: (error) => {
        debugLog('SSE ERROR', error);
        setMessages((prev) => {
          const lastMessage = prev[prev.length - 1];
          if (!lastMessage || lastMessage.role !== 'assistant') return prev;
          return [...prev.slice(0, -1), { ...lastMessage, status: 'done' as const }];
        });
        setStatus('error');
      },
      onMessage: (message) => {
        const { data } = message;
        if (!data) return;

        // Chat completion chunks (OpenAI-style streaming)
        if (data.object === 'chat.completion.chunk' && Array.isArray(data.choices)) {
          const { id, model, created, choices } = data;
          const modelLower = typeof model === 'string' ? model.toLowerCase() : '';

          // If any delta has content/tool_calls we are actively generating
          if (
            choices.some(
              (choice: any) =>
                choice?.delta?.content || (choice?.delta?.tool_calls || []).length > 0,
            )
          ) {
            setStatus('generating');
          }

          setMessages((prev) => {
            let next = prev;

            // Ensure we have an assistant message to mutate
            const ensureAssistant = (role = 'assistant'): ChatMessage => {
              const last = next[next.length - 1];
              if (last && last.role === role) {
                return last;
              }
              const createdMessage: ChatMessage = {
                id: id || (typeof crypto !== 'undefined' && crypto.randomUUID
                  ? crypto.randomUUID()
                  : `${Date.now()}`),
                role: role as ChatMessageRole,
                status: 'thinking',
                raw: '',
                content: '',
                reasoning: '',
                thinking: '',
                toolCalls: [],
                toolOutputs: [],
                sources: [],
                traceRaw: '',
                createdAt: created || performance.now(),
              };
              next = [...next, createdMessage];
              return createdMessage;
            };

            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            choices.forEach((choice: any) => {
              const delta = choice?.delta || {};
              const role: ChatMessageRole = delta.role || 'assistant';
              const contentDelta: string | undefined = extractTextDelta(delta.content);
              const toolCallsDelta: any[] | undefined = delta.tool_calls;
              const finishReason: string | undefined = choice?.finish_reason;

              let assistant = ensureAssistant(role);

              // Merge tool calls
              if (Array.isArray(toolCallsDelta) && toolCallsDelta.length > 0) {
                const updatedToolCalls: ToolCall[] = [...(assistant.toolCalls || [])];
                toolCallsDelta.forEach((toolCall) => {
                  const callId = toolCall?.id || crypto.randomUUID();
                  const fnName = toolCall?.function?.name || '';
                  const argsDelta = toolCall?.function?.arguments || '';
                  const existingIndex = updatedToolCalls.findIndex((c) => c.id === callId);
                  if (existingIndex >= 0) {
                    const existing = updatedToolCalls[existingIndex];
                    updatedToolCalls[existingIndex] = {
                      ...existing,
                      name: existing.name || fnName,
                      arguments: (existing.arguments || '') + argsDelta,
                      status: 'pending' as const,
                    };
                  } else {
                    updatedToolCalls.push({
                      id: callId,
                      name: fnName,
                      arguments: argsDelta,
                      status: 'pending' as const,
                    });
                  }
                });
                assistant = { ...assistant, toolCalls: updatedToolCalls };
              }

              // Merge sources if present in delta
              const deltaSources = (delta as any).sources;
              if (deltaSources && (Array.isArray(deltaSources) || typeof deltaSources === 'string')) {
                const incoming = parseSourcesFromPayload(deltaSources);
                assistant = {
                  ...assistant,
                  sources: mergeSources(assistant.sources, incoming),
                };
              }

              // Merge content / reasoning
              if (typeof contentDelta === 'string' && contentDelta.length > 0) {
                const raw = (assistant.raw || '') + contentDelta;
                let reasoning = assistant.reasoning || assistant.thinking || '';
                let content = assistant.content || '';

                if (modelLower.includes('gpt-oss')) {
                  const parsed = parseGenerationGpt(raw);
                  reasoning = parsed.analysis || parsed.thinking || reasoning;
                  content = parsed.final || content || raw;
                } else if (modelLower.includes('qwen')) {
                  const parsed = parseGenerationQwen(raw);
                  reasoning = parsed.think || reasoning;
                  content = parsed.content || content || raw;
                } else {
                  // Heuristic: if "Search & Reasoning Trace" appears, stash it as traceRaw
                  if (raw.includes('Search & Reasoning Trace')) {
                    const [, traceAndRest = ''] = raw.split(/Search & Reasoning Trace:/);
                    const [maybeTrace, maybeAnswer] = traceAndRest.split(/Answer:/i);
                    assistant = { ...assistant, traceRaw: maybeTrace.trim() };
                    content = (maybeAnswer && maybeAnswer.trim()) || raw;
                  } else {
                    content = raw;
                  }
                }

                assistant = {
                  ...assistant,
                  raw,
                  reasoning,
                  thinking: reasoning,
                  content,
                  status: content ? 'generating' : 'thinking',
                };
              }

              if (finishReason === 'stop') {
                assistant = { ...assistant, status: 'done' };
              }
              if (finishReason === 'tool_calls') {
                assistant = { ...assistant, status: 'thinking' };
              }

              // Replace last assistant
              const last = next[next.length - 1];
              if (last && last.id === assistant.id) {
                next = [...next.slice(0, -1), assistant];
              } else {
                next = [...next, assistant];
              }
            });

            return next;
          });
          // Top-level sources on chunk payload
          if (Array.isArray((data as any).sources)) {
            const incoming = parseSourcesFromPayload((data as any).sources);
            if (incoming.length) {
              setMessages((prev) => {
                if (!prev.length) return prev;
                const last = prev[prev.length - 1];
                if (last.role !== 'assistant') return prev;
                return [
                  ...prev.slice(0, -1),
                  { ...last, sources: mergeSources(last.sources, incoming) },
                ];
              });
            }
          }
          return;
        }

        // Tool outputs (OpenAI style: role "tool" with tool_call_id)
        if (data.role === 'tool') {
          const { tool_call_id: toolCallId, content } = data;
          if (!toolCallId) return;
          setMessages((prev) => {
            const next = [...prev];
            const lastAssistantIndex = [...next].reverse().findIndex((m) => m.role === 'assistant');
            const targetIndex =
              lastAssistantIndex >= 0 ? next.length - 1 - lastAssistantIndex : -1;
            if (targetIndex < 0) return prev;
            const assistant = next[targetIndex];
            const toolOutputs: ToolOutput[] = [...(assistant.toolOutputs || [])];
            const existingIndex = toolOutputs.findIndex((t) => t.toolCallId === toolCallId);
            if (existingIndex >= 0) {
              toolOutputs[existingIndex] = {
                ...toolOutputs[existingIndex],
                content: `${toolOutputs[existingIndex].content || ''}${content || ''}`,
              };
            } else {
              toolOutputs.push({
                toolCallId,
                content: typeof content === 'string' ? content : JSON.stringify(content),
              });
            }
            const incomingSources = parseSourcesFromPayload(content);
            const mergedSources = mergeSources(assistant.sources, incomingSources);
            const toolCalls = (assistant.toolCalls || []).map((call) =>
              call.id === toolCallId ? { ...call, status: 'responded' as const } : call,
            );
            next[targetIndex] = { ...assistant, toolOutputs, toolCalls, sources: mergedSources };
            return next;
          });
          return;
        }
      },
    }),
  );

  const generate = useRefCallback<ChatActions['generate']>((message) => {
    if (clusterStatus !== 'available' || status === 'opened' || status === 'generating') {
      return;
    }

    if (!modelName) {
      return;
    }

    let nextMessages: readonly ChatMessage[] = messages;
    if (message) {
      // Regenerate
      const finalMessageIndex = messages.findIndex((m) => m.id === message.id);
      const finalMessage = messages[finalMessageIndex];
      if (!finalMessage) {
        return;
      }
      nextMessages = nextMessages.slice(
        0,
        finalMessageIndex + (finalMessage.role === 'user' ? 1 : 0),
      );
      debugLog('generate', 'regenerate', nextMessages);
    } else {
      // Generate for new input
      const finalInput = input.trim();
      if (!finalInput) {
        return;
      }
      setInput('');
      const now = performance.now();
      nextMessages = [
        ...nextMessages,
        { id: now.toString(), role: 'user', status: 'done', content: finalInput, createdAt: now },
      ];
      debugLog('generate', 'new', nextMessages);
    }
    setMessages(nextMessages);

    sse.connect(
      modelName,
      nextMessages.map(({ id, role, content }) => ({ id, role, content })),
    );
  });

  const stop = useRefCallback<ChatActions['stop']>(() => {
    debugLog('stop', 'status', status);
    if (status === 'closed' || status === 'error') {
      return;
    }
    sse.disconnect();
  });

  const clear = useRefCallback<ChatActions['clear']>(() => {
    debugLog('clear', 'status', status);
    stop();
    if (status === 'opened' || status === 'generating') {
      return;
    }
    setMessages([]);
  });

  const actions = useConst<ChatActions>({
    setInput,
    generate,
    stop,
    clear,
  });

  const value = useMemo<readonly [ChatStates, ChatActions]>(
    () => [
      {
        input,
        status,
        messages,
      },
      actions,
    ],
    [input, status, messages, actions],
  );

  return <context.Provider value={value}>{children}</context.Provider>;
};

const context = createContext<readonly [ChatStates, ChatActions] | undefined>(undefined);

export const useChat = (): readonly [ChatStates, ChatActions] => {
  const value = useContext(context);
  if (!value) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return value;
};

// ================================================================
// SSE

interface SSEOptions {
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: Error) => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onMessage?: (message: { event: string; id?: string; data: any }) => void;
}

interface RequestMessage {
  readonly id: string;
  readonly role: ChatMessageRole;
  readonly content: string;
}

const createSSE = (options: SSEOptions) => {
  const { onOpen, onClose, onError, onMessage } = options;

  const decoder = new TextDecoder();
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
  let abortController: AbortController | undefined;

  const connect = (model: string, messages: readonly RequestMessage[]) => {
    abortController = new AbortController();
    const url = `${getApiBaseUrl()}/v1/chat/completions`;

    onOpen?.();

    fetch(url, {
      method: 'POST',
      body: JSON.stringify({
        stream: true,
        model,
        messages,
        max_tokens: 2048,
        sampling_params: {
          top_k: 3,
        },
      }),
      signal: abortController.signal,
    })
      .then(async (response) => {
        const statusCode = response.status;
        const contentType = response.headers.get('Content-Type');
        if (statusCode !== 200) {
          onError?.(new Error(`[SSE] Failed to connect: ${statusCode}`));
          return;
        }
        if (!contentType?.includes('text/event-stream')) {
          onError?.(new Error(`[SSE] Invalid content type: ${contentType}`));
          return;
        }

        reader = response.body?.getReader();
        if (!reader) {
          onError?.(new Error(`[SSE] Failed to get reader`));
          return;
        }

        let buffer = '';

        const processLines = (lines: string[]) => {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const message: { event: string; id?: string; data: any } = {
            event: 'message',
            data: undefined,
          };
          lines.forEach((line) => {
            const colonIndex = line.indexOf(':');
            if (colonIndex <= 0) {
              // No colon, skip
              return;
            }

            const field = line.slice(0, colonIndex).trim();
            const value = line.slice(colonIndex + 1).trim();

            if (value.startsWith(':')) {
              // Comment line
              return;
            }

            switch (field) {
              case 'event':
                message.event = value;
                break;
              case 'id':
                message.id = value;
                break;
              case 'data':
                try {
                  // Try to parse as JSON object
                  const data = JSON.parse(value);
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  const walk = (data: any) => {
                    if (!data) {
                      return;
                    }
                    if (Array.isArray(data)) {
                      data.forEach((item, i) => {
                        if (item === null) {
                          data[i] = undefined;
                        } else {
                          walk(item);
                        }
                      });
                    } else if (typeof data === 'object') {
                      Object.keys(data).forEach((key) => {
                        if (data[key] === null) {
                          delete data[key];
                        } else {
                          walk(data[key]);
                        }
                      });
                    }
                  };
                  walk(data);
                  message.data = data;
                } catch (error) {
                  // Parse failed, use original data
                  message.data = value;
                }
                break;
            }

            if (message.data !== undefined) {
              onMessage?.(message);
            }
          });
        };

        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            onClose?.();
            return;
          }

          const chunk = decoder.decode(value);
          buffer += chunk;

          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          processLines(lines);
        }
      })
      .catch((error: Error) => {
        if (error instanceof Error && error.name === 'AbortError') {
          onClose?.();
          return;
        }
        onError?.(error);
      });
  };

  const disconnect = () => {
    reader?.cancel();
    reader = undefined;
    abortController?.abort('stop');
    abortController = undefined;

    onClose?.();
  };

  return { connect, disconnect };
};
