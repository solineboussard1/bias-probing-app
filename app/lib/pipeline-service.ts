import { retrieveSingleCall, ModelKey } from './apiCaller';
import { AnalysisResult, SelectedParams, ProgressCallback } from '../types/pipeline';
import { generatePrompts } from './pipeline';
import pLimit from 'p-limit';

export type BatchResults = {
  prompt: string;
  responses: string[];
  metadata: {
    perspective: string;
    demographics: string[];
    context: string;
    questionType: string;
  };
};

function generateDemographicGroups(demographics: SelectedParams['demographics']): string[][] {
  const groups: string[][] = [];

  // Baseline group: no demographics
  groups.push([]);

  // For each demographic attribute, add one group per selected value
  if (demographics.genders?.length) {
    demographics.genders.forEach(g => groups.push([g]));
  }
  if (demographics.ages?.length) {
    demographics.ages.forEach(a => groups.push([a]));
  }
  if (demographics.ethnicities?.length) {
    demographics.ethnicities.forEach(e => groups.push([e]));
  }
  if (demographics.socioeconomic?.length) {
    demographics.socioeconomic.forEach(s => groups.push([s]));
  }

  return groups;
}

export async function processBatch(
  prompts: string[],
  params: SelectedParams,
  userApiKeys: Record<'openai' | 'anthropic' | 'huggingface' | 'deepseek'|'mistral', string>,
  batchSize: number = params.iterations,
  onProgress?: ProgressCallback
): Promise<BatchResults[]> {
  const results: BatchResults[] = [];
  let completedPrompts = 0;
  const MAX_RESPONSE_SIZE = 1024 * 1024; // 1MB limit per response

  const demographicGroups = params.domain === 'custom'
    ? [[]]
    : generateDemographicGroups(params.demographics);

  const overallLimit = pLimit(5);
  const tasks: Promise<void>[] = [];

  for (const promptTemplate of prompts) {
    for (const demographics of demographicGroups) {
      tasks.push(overallLimit(async () => {
        let prompt = '';
        const responses: string[] = [];

        if (promptTemplate.includes('{demographic}')) {
          prompt = promptTemplate.replace(
            /\{demographic\}/g,
            () => {
              const demo = demographics.join(' ');
              let perspective = 'Hypothetical';
              if (promptTemplate.includes('I am')) perspective = 'First';
              else if (promptTemplate.includes('My friend')) perspective = 'Third';

              if (!demo) return '';
              if (perspective === 'First') return `I am ${demo}.`;
              if (perspective === 'Third') return `My friend is ${demo}.`;
              return `Someone is ${demo}.`;
            }
          );
        } else {
          // baseline or custom‐baked prompt
          let perspective = 'Hypothetical';
          if (promptTemplate.startsWith('I am')) perspective = 'First';
          else if (promptTemplate.startsWith('My friend')) perspective = 'Third';

          const demo = demographics.join(' ');
          let demoPhrase = '';
          if (demo) {
            if (perspective === 'First') demoPhrase = `I am ${demo}. `;
            else if (perspective === 'Third') demoPhrase = `My friend is ${demo}. `;
            else demoPhrase = `Someone is ${demo}. `;
          }
          prompt = demoPhrase + promptTemplate;
        }

        completedPrompts++;
        onProgress?.({
          type: 'prompt-execution',
          message: `Processing prompt ${completedPrompts}/${prompts.length * demographicGroups.length}`,
          prompt: prompt.slice(0, 100) + (prompt.length > 100 ? '...' : ''),
          completedPrompts,
          totalPrompts: prompts.length * demographicGroups.length
        });

        const innerLimit = pLimit(3);
        const iterationTasks: Promise<void>[] = [];
        for (let i = 0; i < batchSize; i++) {
          iterationTasks.push(innerLimit(async () => {
            try {
              const response = await retrieveSingleCall(prompt, params.model as ModelKey, userApiKeys);
              const sanitized = response && response.length < MAX_RESPONSE_SIZE
                ? response.replace(/[\u0000-\u001F\u007F-\u009F]/g, '').trim()
                : 'Response too large or empty';
              responses.push(sanitized);
            } catch {
              responses.push('Failed to get response');
            }
          }));
        }
        await Promise.all(iterationTasks);

        let demoMeta: string[];
        if (params.domain === 'custom') {
          const m = prompt.match(/^(?:I am|My friend is|Someone is) ([^.]+)\./);
          demoMeta = m ? [m[1]] : ['baseline'];
        } else {
          demoMeta = demographics.length > 0
            ? demographics.map(d => d.slice(0, 100))
            : ['baseline'];
        }

        const perspectiveMeta = prompt.startsWith('I am')
          ? 'First'
          : prompt.startsWith('My friend')
            ? 'Third'
            : 'Hypothetical';

        const questionMeta = params.questionTypes.find(qt => prompt.includes(qt)) || 'Unknown';

        results.push({
          prompt: prompt.slice(0, 1000),
          responses,
          metadata: {
            perspective: perspectiveMeta,
            demographics: demoMeta,
            context: params.context.slice(0, 1000),
            questionType: questionMeta
          }
        });
      }));
    }
  }

  await Promise.all(tasks);
  return results;
}

export async function runAnalysisPipeline(
  params: SelectedParams,
  userApiKeys: Record<'openai' | 'anthropic' | 'huggingface' | 'deepseek'|'mistral', string>,
  onProgress?: ProgressCallback
): Promise<AnalysisResult> {
  onProgress?.({ type: 'prompt-generation', message: 'Generating prompts...' });

  const promptTemplates = generatePrompts(params);
  const demographicGroups = generateDemographicGroups(params.demographics);

  onProgress?.({
    type: 'prompt-generation',
    message: `Generated ${promptTemplates.length} prompt templates with ${demographicGroups.length} demographic groups each`,
    totalPrompts: promptTemplates.length * demographicGroups.length
  });

  const batchResults = await processBatch(
    promptTemplates,
    params,
    userApiKeys,
    params.iterations,
    onProgress
  );

  return {
    id: crypto.randomUUID(),
    modelName: params.model,
    concept: params.primaryIssues.join(', '),
    demographics: [
      ...params.demographics.genders,
      ...params.demographics.ages,
      ...params.demographics.ethnicities,
      ...params.demographics.socioeconomic
    ],
    context: params.context,
    details: `Analyzed ${promptTemplates.length * demographicGroups.length} prompts (${promptTemplates.length} templates × ${demographicGroups.length} demographic groups)`,
    timestamp: new Date().toISOString(),
    prompts: batchResults.map(br => ({
      text: br.prompt,
      responses: br.responses,
      metadata: br.metadata
    }))
  };
}