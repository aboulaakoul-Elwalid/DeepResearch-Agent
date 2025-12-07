import { DrawerLayout } from '../components/common';
import { ChatInput, ChatMessages } from '../components/inputs';
import {
  Box,
  Stack,
  Typography,
  Divider,
  Paper,
  IconButton,
  Button,
  useMediaQuery,
  useTheme,
  TextField,
} from '@mui/material';
import { IconLayoutSidebarRightCollapse, IconLayoutSidebarRightExpand, IconPlus } from '@tabler/icons-react';
import { useChat, useCluster, getApiBaseUrl, setApiBaseUrl } from '../services';
import { useState } from 'react';

const useLatestAssistant = () => {
  const [{ messages }] = useChat();
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'assistant') return messages[i];
  }
  return undefined;
};

export default function PageChat() {
  const theme = useTheme();
  const isSmall = useMediaQuery(theme.breakpoints.down('md'));
  const latestAssistant = useLatestAssistant();
  const [, { clear }] = useChat();
  const [, { refreshApiBase }] = useCluster();
  const [traceOpen, setTraceOpen] = useState(!isSmall);
  const [apiBaseUrlInput, setApiBaseUrlInput] = useState(getApiBaseUrl());

  return (
    <DrawerLayout>
      <Stack direction={{ xs: 'column', md: 'row' }} sx={{ width: '100%', height: '100%', gap: 3 }}>
        <Stack flex={2} sx={{ gap: 2, minHeight: 0 }}>
          <Stack direction='row' justifyContent='space-between' alignItems='center'>
            <Button
              size='small'
              color='primary'
              startIcon={<IconPlus size='1rem' />}
              onClick={clear}
              disabled={latestAssistant === undefined}
            >
              New chat
            </Button>
            <IconButton
              size='small'
              onClick={() => setTraceOpen((v) => !v)}
              aria-label={traceOpen ? 'Hide trace panel' : 'Show trace panel'}
            >
              {traceOpen ? <IconLayoutSidebarRightCollapse /> : <IconLayoutSidebarRightExpand />}
            </IconButton>
          </Stack>
          <ChatMessages />
          <ChatInput />
        </Stack>
        {traceOpen && (
          <Paper
            variant='outlined'
            sx={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              gap: 2,
              minHeight: 0,
              p: 2,
              bgcolor: 'grey.50',
            }}
          >
            <Typography variant='subtitle1'>Research Trace</Typography>
            <Divider />
            <Box sx={{ flex: 1, overflow: 'auto' }}>
              {latestAssistant?.toolCalls?.length ? (
                <Stack gap={1.5}>
                  {latestAssistant.toolCalls.map((call) => (
                    <Box key={call.id} sx={{ p: 1.25, borderRadius: 2, border: '1px solid', borderColor: 'grey.200' }}>
                      <Typography variant='subtitle2'>{call.name || 'tool_call'}</Typography>
                      <Typography variant='caption' color='text.secondary' sx={{ whiteSpace: 'pre-wrap' }}>
                        {call.arguments}
                      </Typography>
                      {latestAssistant.toolOutputs
                        ?.filter((output) => output.toolCallId === call.id)
                        .map((output) => (
                          <Box key={output.toolCallId} sx={{ mt: 0.75 }}>
                            <Typography variant='caption' color='text.secondary'>
                              Output
                            </Typography>
                            <Typography variant='body2' sx={{ whiteSpace: 'pre-wrap' }}>
                              {output.summary || output.content}
                            </Typography>
                          </Box>
                        ))}
                    </Box>
                  ))}
                </Stack>
              ) : (
                <Typography variant='body2' color='text.secondary'>
                  Tool calls and traces will appear here.
                </Typography>
              )}
            </Box>
            <Divider />
            <Typography variant='subtitle1'>Sources</Typography>
            <Box sx={{ flex: 1, overflow: 'auto', display: 'flex', flexDirection: 'column', gap: 1 }}>
              {latestAssistant?.sources?.length ? (
                latestAssistant.sources.map((source, idx) => (
                  <Paper key={source.id || idx} variant='outlined' sx={{ p: 1.25, borderRadius: 2 }}>
                    <Typography variant='subtitle2'>
                      {idx + 1}. {source.title || source.url || 'Source'}
                    </Typography>
                    {source.url && (
                      <Typography
                        variant='caption'
                        color='primary'
                        component='a'
                        href={source.url}
                        target='_blank'
                        rel='noreferrer'
                        sx={{ display: 'block', mb: 0.5, wordBreak: 'break-all' }}
                      >
                        {source.url}
                      </Typography>
                    )}
                    {source.snippet && (
                      <Typography variant='body2' color='text.secondary'>
                        {source.snippet}
                      </Typography>
                    )}
                  </Paper>
                ))
              ) : (
                <Typography variant='body2' color='text.secondary'>
                  Citations from the assistant will show up here.
                </Typography>
              )}
            </Box>
            <Divider />
            <Typography variant='subtitle1'>Endpoint</Typography>
            <Stack direction='row' gap={1} alignItems='center'>
              <TextField
                size='small'
                fullWidth
                label='API base URL'
                value={apiBaseUrlInput}
                onChange={(e) => setApiBaseUrlInput(e.target.value)}
              />
              <Button
                size='small'
                variant='outlined'
                onClick={() => refreshApiBase(apiBaseUrlInput.trim())}
                disabled={apiBaseUrlInput.trim() === getApiBaseUrl()}
              >
                Save
              </Button>
            </Stack>
          </Paper>
        )}
      </Stack>
    </DrawerLayout>
  );
}
