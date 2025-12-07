import { memo, useEffect, useMemo, useRef, useState, type FC, type UIEventHandler } from 'react';
import { useChat, type ChatMessage, type Source, type ToolCall, type ToolOutput } from '../../services';
import {
  Box,
  Chip,
  IconButton,
  Paper,
  Stack,
  Tooltip,
  Typography,
} from '@mui/material';
import {
  IconArrowDown,
  IconCopy,
  IconCopyCheck,
  IconRefresh,
  IconTimeline,
  IconLink,
  IconActivityHeartbeat,
} from '@tabler/icons-react';
import { useRefCallback } from '../../hooks';
import ChatMarkdown from './chat-markdown';
import { DotPulse } from './dot-pulse';

export const ChatMessages: FC = () => {
  const [{ status, messages }] = useChat();

  const refContainer = useRef<HTMLDivElement>(null);
  // const refBottom = useRef<HTMLDivElement>(null);
  const [isBottom, setIsBottom] = useState(true);

  const userScrolledUpRef = useRef(false);
  const autoScrollingRef = useRef(false);
  const prevScrollTopRef = useRef(0);

  const scrollToBottom = useRefCallback(() => {
    const el = refContainer.current;
    if (!el) return;
    userScrolledUpRef.current = false;
    autoScrollingRef.current = true;
    requestAnimationFrame(() => {
      el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
      // el.lastElementChild?.scrollIntoView({ behavior: 'smooth' });
    });
    setTimeout(() => {
      autoScrollingRef.current = false;
    }, 250);
  });

  useEffect(() => {
    if (userScrolledUpRef.current) return;
    autoScrollingRef.current = true;
    scrollToBottom();
    const t = setTimeout(() => {
      autoScrollingRef.current = false;
    }, 200);
    return () => clearTimeout(t);
  }, [messages]);

  const onScroll = useRefCallback<UIEventHandler<HTMLDivElement>>((event) => {
    event.stopPropagation();

    const container = refContainer.current;
    if (!container) return;

    const { scrollTop, scrollHeight, clientHeight } = container;
    const bottomGap = scrollHeight - scrollTop - clientHeight;

    setIsBottom(bottomGap < 10);

    if (!autoScrollingRef.current) {
      if (scrollTop < prevScrollTopRef.current - 2) {
        userScrolledUpRef.current = true;
      }
    }
    prevScrollTopRef.current = scrollTop;

    if (bottomGap < 10) {
      userScrolledUpRef.current = false;
    }
  });

  const nodeScrollToBottomButton = (
    <IconButton
      key='scroll-to-bottom'
      onClick={scrollToBottom}
      size='small'
      aria-label='Scroll to bottom'
      sx={{
        position: 'absolute',
        right: 12,
        bottom: 8,
        width: 28,
        height: 28,
        bgcolor: 'white',
        border: '1px solid',
        borderColor: 'grey.300',
        '&:hover': { bgcolor: 'grey.100' },
        opacity: isBottom ? 0 : 1,
        pointerEvents: isBottom ? 'none' : 'auto',
        transition: 'opacity .15s ease',
      }}
    >
      <IconArrowDown />
    </IconButton>
  );

  const nodeStream = (
    <Stack
      key='stream'
      ref={refContainer}
      sx={{
        width: '100%',
        height: '100%',

        overflowX: 'hidden',
        overflowY: 'scroll',
        '&::-webkit-scrollbar': { display: 'none' },
        scrollbarWidth: 'none',
        msOverflowStyle: 'none',

        display: 'flex',
        gap: 4,
      }}
      onScroll={onScroll}
      onWheel={(e) => {
        if (e.deltaY < 0) userScrolledUpRef.current = true;
      }}
      onTouchMove={() => {
        userScrolledUpRef.current = true;
      }}
    >
      {messages.map((message, idx) => (
        <ChatMessage key={message.id} message={message} isLast={idx === messages.length - 1} />
      ))}

      {status === 'opened' && <DotPulse size='large' />}

      {/* Last child for scroll to bottom */}
      <Box sx={{ width: '100%', height: 0 }} />
    </Stack>
  );

  return (
    <Box
      sx={{
        position: 'relative',
        flex: 1,
        overflow: 'hidden',
      }}
    >
      {nodeStream}
      {nodeScrollToBottomButton}
    </Box>
  );
};

const ChatMessage: FC<{ message: ChatMessage; isLast?: boolean }> = memo(({ message, isLast }) => {
  const {
    role,
    status: messageStatus,
    thinking,
    reasoning,
    content,
    toolCalls = [],
    toolOutputs = [],
    sources = [],
    traceRaw,
  } = message;

  const [, { generate }] = useChat();

  const [copied, setCopied] = useState(false);
  useEffect(() => {
    const timeoutId = setTimeout(() => setCopied(false), 2000);
    return () => clearTimeout(timeoutId);
  }, [copied]);

  const onCopy = useRefCallback(() => {
    navigator.clipboard.writeText(content);
    setCopied(true);
  });

  const onRegenerate = useRefCallback(() => {
    generate(message);
  });

  const justifyContent = role === 'user' ? 'flex-end' : 'flex-start';

  const finalReasoning = reasoning || thinking;
  const nodeContent =
    role === 'user' ?
      <Typography
        key='user-message'
        variant='body1'
        sx={{
          px: 2,
          py: 1.5,
          borderRadius: '0.5rem',
          backgroundColor: 'background.default',
          fontSize: '0.9rem',
          whiteSpace: 'pre-wrap',
        }}
      >
        {content}
      </Typography>
    : (
      <Stack key='assistant-message-stack' gap={1.5}>
        {finalReasoning && (
          <Paper variant='outlined' sx={{ p: 1.5, borderRadius: 2, bgcolor: 'grey.50' }}>
            <Stack direction='row' alignItems='center' gap={0.75} sx={{ mb: 0.5 }}>
              <IconActivityHeartbeat size='1rem' />
              <Typography variant='caption' color='text.secondary'>
                Reasoning
              </Typography>
            </Stack>
            <ChatMarkdown isThinking content={finalReasoning} />
          </Paper>
        )}

        {!!toolCalls.length && (
          <Paper variant='outlined' sx={{ p: 1.25, borderRadius: 2 }}>
            <Stack direction='row' alignItems='center' gap={0.75} sx={{ mb: 0.5 }}>
              <IconTimeline size='1rem' />
              <Typography variant='caption' color='text.secondary'>
                Tool timeline
              </Typography>
            </Stack>
            <TraceTimeline toolCalls={toolCalls} toolOutputs={toolOutputs} />
          </Paper>
        )}

        {sources.length > 0 && <SourceChips sources={sources} />}

        {content && (
          <Paper variant='outlined' sx={{ p: 1.5, borderRadius: 2 }}>
            <ChatMarkdown key='assistant-message' content={content} />
          </Paper>
        )}

        {traceRaw && (
          <Paper variant='outlined' sx={{ p: 1.25, borderRadius: 2, bgcolor: 'grey.50' }}>
            <Typography variant='caption' color='text.secondary'>
              Raw trace
            </Typography>
            <Typography variant='body2' sx={{ whiteSpace: 'pre-wrap' }}>
              {traceRaw}
            </Typography>
          </Paper>
        )}
      </Stack>
    );

  const assistantDone = messageStatus === 'done';
  const showCopy = role === 'user' || (role === 'assistant' && assistantDone);
  const showRegenerate = role === 'assistant' && assistantDone;

  const userHoverRevealSx =
    role === 'user' ?
      {
        '&:hover .actions-user': {
          opacity: 1,
          pointerEvents: 'auto',
        },
      }
    : {};

  return (
    <Stack direction='row' sx={{ width: '100%', justifyContent }}>
      <Stack
        sx={{
          maxWidth: role === 'user' ? { xs: '100%', md: '80%' } : '100%',
          alignSelf: role === 'user' ? 'flex-end' : 'flex-start',
          gap: 1,
          ...userHoverRevealSx,
        }}
      >
        {nodeContent}

        {(showCopy || showRegenerate) && (
          <Stack
            key='actions'
            direction='row'
            className={role === 'user' ? 'actions-user' : undefined}
            sx={{
              justifyContent,
              color: 'grey.600',
              gap: 0.5,
              ...(role === 'user' ?
                {
                  opacity: 0,
                  pointerEvents: 'none',
                  transition: 'opacity .15s ease',
                }
              : {}),
            }}
          >
            {showCopy && (
              <Tooltip
                key='copy'
                title={copied ? 'Copied!' : 'Copy'}
                slotProps={{
                  tooltip: { sx: { bgcolor: 'primary.main', borderRadius: 1 } },
                  popper: { modifiers: [{ name: 'offset', options: { offset: [0, -8] } }] },
                }}
              >
                <IconButton
                  onClick={onCopy}
                  size='small'
                  sx={{
                    width: 24,
                    height: 24,
                    borderRadius: '8px',
                    '&:hover': { bgcolor: 'action.hover' },
                  }}
                >
                  {copied ?
                    <IconCopyCheck />
                  : <IconCopy />}
                </IconButton>
              </Tooltip>
            )}

            {showRegenerate && (
              <Tooltip
                key='regenerate'
                title='Regenerate'
                slotProps={{
                  tooltip: { sx: { bgcolor: 'primary.main', borderRadius: 1 } },
                  popper: { modifiers: [{ name: 'offset', options: { offset: [0, -8] } }] },
                }}
              >
                <IconButton
                  onClick={onRegenerate}
                  size='small'
                  sx={{
                    width: 24,
                    height: 24,
                    borderRadius: '8px',
                    '&:hover': { bgcolor: 'action.hover' },
                  }}
                >
                  <IconRefresh />
                </IconButton>
              </Tooltip>
            )}
          </Stack>
        )}
      </Stack>
    </Stack>
  );
});

const TraceTimeline: FC<{ toolCalls: readonly ToolCall[]; toolOutputs: readonly ToolOutput[] }> = ({
  toolCalls,
  toolOutputs,
}) => {
  const outputsByCall = useMemo(() => {
    const map: Record<string, ToolOutput> = {};
    toolOutputs.forEach((o) => {
      map[o.toolCallId] = o;
    });
    return map;
  }, [toolOutputs]);

  return (
    <Stack gap={1}>
      {toolCalls.map((call) => {
        const output = outputsByCall[call.id];
        const status = call.status || (output ? 'responded' : 'pending');
        const statusLabel =
          status === 'responded' ? 'done' : status === 'called' ? 'called' : 'pending';
        return (
          <Stack key={call.id} gap={0.5} sx={{ borderLeft: '2px solid', borderColor: 'grey.200', pl: 1 }}>
            <Stack direction='row' gap={0.75} alignItems='center'>
              <Typography variant='subtitle2' sx={{ fontWeight: 600 }}>
                {call.name || 'tool_call'}
              </Typography>
              <Chip size='small' label={statusLabel} color={status === 'responded' ? 'success' : 'default'} />
            </Stack>
            {call.arguments && (
              <Typography variant='caption' color='text.secondary' sx={{ whiteSpace: 'pre-wrap' }}>
                {call.arguments}
              </Typography>
            )}
            {output && (
              <Box sx={{ pl: 0.5 }}>
                <Typography variant='caption' color='text.secondary'>
                  Output
                </Typography>
                <Typography variant='body2' sx={{ whiteSpace: 'pre-wrap' }}>
                  {output.summary || output.content}
                </Typography>
              </Box>
            )}
          </Stack>
        );
      })}
    </Stack>
  );
};

const SourceChips: FC<{ sources: readonly Source[] }> = ({ sources }) => {
  if (!sources.length) return null;
  return (
    <Stack direction='row' gap={0.5} flexWrap='wrap'>
      {sources.map((source, idx) => (
        <Chip
          key={source.id || idx}
          size='small'
          icon={<IconLink size='0.9rem' />}
          label={
            source.title
              ? `${idx + 1}. ${source.title}`
              : `${idx + 1}. ${source.url || 'source'}`
          }
          component='a'
          href={source.url}
          target={source.url ? '_blank' : undefined}
          rel='noreferrer'
          clickable={!!source.url}
        />
      ))}
    </Stack>
  );
};
