import { createHttpStreamFactory } from './http-stream';
import { getApiBaseUrl } from './config';

export const getModelList = async (): Promise<readonly any[]> => {
  const response = await fetch(`${getApiBaseUrl()}/model/list`, { method: 'GET' });
  const message = await response.json();
  if (message.type !== 'model_list') {
    throw new Error(`Invalid message type: ${message.type}.`);
  }
  return message.data;
};

export const initScheduler = async (params: {
  model_name: string;
  init_nodes_num: number;
  is_local_network: boolean;
}): Promise<void> => {
  const response = await fetch(`${getApiBaseUrl()}/scheduler/init`, {
    method: 'POST',
    body: JSON.stringify(params),
  });
  const message = await response.json();
  if (message.type !== 'scheduler_init') {
    throw new Error(`Invalid message type: ${message.type}.`);
  }
  return message.data;
};

export const createStreamClusterStatus = (
  options: Parameters<ReturnType<typeof createHttpStreamFactory>>[0],
) =>
  createHttpStreamFactory({
    url: `${getApiBaseUrl()}/cluster/status`,
    method: 'GET',
  })(options);
