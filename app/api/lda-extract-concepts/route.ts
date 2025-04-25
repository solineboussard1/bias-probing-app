import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import { AnalysisResult, LDAResult } from '@/app/types/pipeline';

export async function POST(req: Request): Promise<Response> {
  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      try {
        const results: AnalysisResult[] = await req.json();
        const responses = createResponseData(results);

        sendSSE(controller, encoder, {
          type: 'extraction_progress',
          message: 'Starting LDA concept extraction',
          progress: { processed: 0, total: responses.length },
        });

        const pythonScript = path.join(process.cwd(), 'app', 'python', 'lda_extractor.py');
        const pythonProcess = spawn('python', [pythonScript]);

        pythonProcess.stdin.write(JSON.stringify(responses));
        pythonProcess.stdin.end();

        let outputData = '';

        pythonProcess.stdout.on('data', (data) => {
          outputData += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
          const errorStr = data.toString();
          console.error(`Python Error: ${errorStr}`);
          if (errorStr.toLowerCase().includes('error')) {
            sendSSE(controller, encoder, { type: 'error', error: errorStr });
          }
        });

        pythonProcess.on('close', (code) => {
          if (code === 0) {
            try {
              const ldaResults: LDAResult = JSON.parse(outputData);
              if (ldaResults.error) {
                sendSSE(controller, encoder, { type: 'error', error: ldaResults.error });
              } else {
                sendSSE(controller, encoder, {
                  type: 'lda_concepts',
                  topics: ldaResults.topics,
                  distributions: ldaResults.distributions,
                  demographicDistributions: ldaResults.demographicDistributions,
                  progress: { processed: responses.length, total: responses.length },
                });
                sendSSE(controller, encoder, { type: 'complete', message: 'LDA concept extraction completed' });
              }
            } catch {
              sendSSE(controller, encoder, { type: 'error', message: 'Failed to parse Python output' });
            }
          } else {
            sendSSE(controller, encoder, { type: 'error', message: 'Python process exited with non-zero code' });
          }
          controller.close();
        });
      } catch (error) {
        console.error('Concept extraction failed:', error);
        sendSSE(controller, encoder, {
          type: 'error',
          message: error instanceof Error ? error.message : 'Unknown error',
        });
        controller.close();
      }
    },
  });

  return new NextResponse(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
}

function sendSSE(controller: ReadableStreamDefaultController, encoder: TextEncoder, data: object) {
  controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
}

function createResponseData(results: AnalysisResult[]): { text: string; demographics: { category: string; value: string }[] }[] {
  return results.flatMap(result =>
    result.prompts.flatMap(prompt => {
      const demographics = extractDemographics(prompt.metadata.demographics);
      return prompt.responses.map(response => ({
        text: response,
        demographics,
      }));
    })
  );
}

function extractDemographics(demos?: string[]): { category: string; value: string }[] {
  if (!demos || demos.length === 0) return [{ category: 'baseline', value: 'baseline' }];
  if (demos.length === 1 && demos[0].toLowerCase() === 'baseline') return [{ category: 'baseline', value: 'baseline' }];
  return demos.map(demo => {
    const demoLower = demo.toLowerCase();
    const category = getDemographicCategory(demoLower) || 'all';
    return { category, value: demo };
  });
}

function getDemographicCategory(demo: string): string | null {
  if (['woman', 'man', 'non-binary'].includes(demo)) return 'genders';
  if (['young adult', 'middle-aged', 'elderly'].includes(demo)) return 'ages';
  if (['asian', 'black', 'hispanic', 'white', 'other'].includes(demo)) return 'ethnicities';
  if (['low income', 'middle income', 'high income'].includes(demo)) return 'socioeconomic';
  return null;
}
