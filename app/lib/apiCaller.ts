import OpenAI from 'openai';
import { InferenceClient } from '@huggingface/inference';
import Anthropic from '@anthropic-ai/sdk';
import { Mistral } from '@mistralai/mistralai';
import https from 'https';

const delay = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms));

interface ModelSettings {
  provider: 'openai' | 'anthropic' | 'huggingface' | 'deepseek' | 'mistral';
  modelName: string;
  endpoint?: string;
}

const modelConfig: Record<string, ModelSettings> = {
  'gpt-4o': {
    provider: 'openai',
    modelName: 'gpt-4',
    endpoint: 'https://api.openai.com/v1/chat/completions',
  },
  'gpt-4o-mini': {
    provider: 'openai',
    modelName: 'gpt-3.5-turbo',
    endpoint: 'https://api.openai.com/v1/chat/completions',
  },
  'gpt-o1-mini': {
    provider: 'openai',
    modelName: 'gpt-o1-mini',
    endpoint: 'https://api.openai.com/v1/chat/completions',
  },
  'claude-3-5-sonnet-latest': {
    provider: 'anthropic',
    modelName: 'claude-3-5-sonnet-latest',
  },
  'mistral-large-latest': {
    provider: 'mistral',
    modelName: 'mistral-large-latest',
  },
  'llama-3-8b': {
    provider: 'huggingface',
    modelName: 'meta-llama/Llama-3-8B-Instruct',
    endpoint: 'https://api-inference.huggingface.co/models/meta-llama/Llama-3-8B-Instruct',
  },
  'deepseek-chat': {
    provider: 'deepseek',
    modelName: 'deepseek-chat',
    endpoint: 'https://api.deepseek.com/v1/chat/completions',
  },
};

export type ModelKey = keyof typeof modelConfig;

function createAgent() {
  return new https.Agent({
    keepAlive: true,
    maxSockets: 10,
    timeout: 60_000,
  });
}

export async function retrieveSingleCall(
  prompt: string,
  selectedModel: ModelKey,
  userApiKeys: Record<'openai' | 'anthropic' | 'huggingface' | 'deepseek' | 'mistral', string>
): Promise<string> {
  const config = modelConfig[selectedModel];
  const userApiKey = userApiKeys[config.provider];
  const maxRetries = 3;
  let attempts = 0;

  while (attempts < maxRetries) {
    try {
      // OpenAI & DeepSeek (OpenAI-compatible)
      if (config.provider === 'openai' || config.provider === 'deepseek') {
        const openai = new OpenAI({
          apiKey: userApiKey,
          httpAgent: createAgent(),
          timeout: 60_000,
        });

        const response = await openai.chat.completions.create({
          model: config.modelName,
          messages: [
            { role: 'system', content: 'You are a helpful assistant.' },
            { role: 'user', content: prompt },
          ],
          temperature: 0.7,
          max_tokens: 500,
        });

        const content = response.choices?.[0]?.message?.content;
        if (!content) throw new Error('No response from OpenAI.');
        return content;
      }

      // Anthropic (Claude)
      else if (config.provider === 'anthropic') {
        const anthropic = new Anthropic({ apiKey: userApiKey });
        const response = await anthropic.messages.create({
          model: config.modelName,
          system: 'You are a helpful assistant.',
          messages: [{ role: 'user', content: prompt }],
          max_tokens: 500,
        });

        let content: string;
        if (typeof response.content === 'string') {
          content = response.content;
        } else if (Array.isArray(response.content)) {
          content = response.content
            .map(block => ('text' in block ? block.text : ''))
            .join(' ')
            .trim();
        } else {
          throw new Error('Invalid response format from Anthropic.');
        }

        if (!content) throw new Error('No response from Anthropic.');
        return content;
      }

      // Mistral AI
      else if (config.provider === 'mistral') {
        const client = new Mistral({ apiKey: userApiKey });
        const resp = await client.chat.complete({
          model: config.modelName,
          messages: [{ role: 'user', content: prompt }],
        });
        const text = resp.choices?.[0]?.message?.content;
        if (!text) throw new Error('No response from Mistral.');

        if (Array.isArray(text)) {
          throw new Error('Invalid response format from Mistral.');
        }

        return text;
      }

      // Hugging Face Inference (Llama)
      else if (config.provider === 'huggingface') {
        const hf = new InferenceClient(userApiKey);
        const result = await hf.textGeneration({
          model: config.modelName,
          inputs: prompt,
          parameters: { max_new_tokens: 500, return_full_text: false },
        });

        if (result && typeof result === 'object' && 'generated_text' in result) {
          return result.generated_text;
        }
        throw new Error('No response from Hugging Face.');
      }

      else {
        throw new Error(`Unsupported provider: ${config.provider}`);
      }
    }
    catch (err: unknown) {
      attempts++;
      if (attempts >= maxRetries) {
        throw new Error(`API request failed after ${maxRetries} attempts: ${err}`);
      }
      await delay(1_000 * attempts);
    }
  }

  throw new Error(`Exhausted retries for model ${selectedModel}`);
}
