import { OpenAI } from 'openai';
import { NextResponse } from 'next/server';
import { AnalysisResult, ExtractedConcepts } from '@/app/types/pipeline';
import { spawn } from 'child_process';
import path from 'path';

async function runConceptClustering(
  allConceptFrequencies: Map<string, number>,
  subgroupConcepts: Map<string, string[]> = new Map(),
  userApiKeys: Record<'openai' | 'anthropic' | 'huggingface' | 'deepseek'|'mistral', string>
) {
  return new Promise((resolve, reject) => {
    const clusteringScript = path.join(process.cwd(), 'app', 'python', 'concept_clustering.py');
    const pythonProcess = spawn('python', [clusteringScript]);
  
    let outputData = '';
    let errorData = '';
  
    pythonProcess.stdout.on('data', (data) => {
      outputData += data.toString();
    });
  
    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
    });
  
    pythonProcess.on('close', (code) => {
      console.log("Python stderr:", errorData);
      if (code !== 0) {
        console.error('Python clustering error:', errorData);
        return reject(new Error(`Clustering failed with code ${code}`));
      }
      try {
        // Parse only the last JSON line, in case there is extra logging
        const outputLines = outputData.trim().split('\n');
        const lastLine = outputLines[outputLines.length - 1];
        const clusters = JSON.parse(lastLine);
        resolve(clusters);
      } catch (error) {
        reject(error);
      }
    });
  
    // Create subgroup frequency mapping
    const subgroupFreqMap = new Map<string, Map<string, number>>();
    for (const [demo, concepts] of subgroupConcepts.entries()) {
      const freqMap = new Map<string, number>();
      concepts.forEach(concept => {
        freqMap.set(concept, (freqMap.get(concept) || 0) + 1);
      });
      subgroupFreqMap.set(demo, freqMap);
    }
  
    const inputData = {
      all: Array.from(allConceptFrequencies.entries()),
      demographics: Object.fromEntries(
        [...subgroupFreqMap.entries()].map(([demo, freqMap]) => [
          demo,
          Array.from(freqMap.entries())
        ])
      ),
      userApiKeys: { huggingface: userApiKeys.huggingface }
    };
  
    console.log("Sending clustering input data:", inputData);
    pythonProcess.stdin.write(JSON.stringify(inputData));
    pythonProcess.stdin.end();
  });
}

export async function POST(req: Request): Promise<Response> {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      try {
        const { results, userApiKeys } = await req.json() as {
          results: AnalysisResult[];
          userApiKeys: Record<'openai' | 'anthropic' | 'huggingface'|'deepseek'|'mistral', string>;
        };

        if (!results || !Array.isArray(results)) {
          return NextResponse.json({ error: "'results' is missing or not an array" }, { status: 400 });
        }
        if (!userApiKeys || typeof userApiKeys !== "object") {
          return NextResponse.json({ error: "'userApiKeys' is missing or invalid" }, { status: 400 });
        }

        let totalResponses = 0, processedResponses = 0;
        results.forEach(result => {
          if (!result.prompts || !Array.isArray(result.prompts)) return;
          result.prompts.forEach(prompt => {
            if (!prompt.responses || !Array.isArray(prompt.responses)) return;
            totalResponses += prompt.responses.length;
          });
        });

        const allConcepts: string[] = [];
        const subgroupConcepts: Map<string, string[]> = new Map();

        for (const result of results) {
          for (const prompt of result.prompts) {
            const demographics = getDemographics(prompt.metadata.demographics);
            for (const response of prompt.responses) {
              processedResponses++;
              controller.enqueue(
                encoder.encode(`data: ${JSON.stringify({
                  type: 'extraction_progress',
                  message: `Extracting concepts from response ${processedResponses}/${totalResponses}`,
                  progress: { processed: processedResponses, total: totalResponses }
                })}\n\n`)
              );

              const concepts = await extractConcepts(response, userApiKeys);
              if (concepts.length > 0) {
                allConcepts.push(...concepts);
                demographics.forEach(demo => {
                  const key = demo.category === 'baseline' ? 'baseline' : `${demo.category}:${demo.value}`;
                  if (!subgroupConcepts.has(key)) subgroupConcepts.set(key, []);
                  subgroupConcepts.get(key)?.push(...concepts);
                });

                const extractedConcept: ExtractedConcepts = {
                  concepts,
                  demographics,
                  response: response.replace(/[\n\r]+/g, ' ').trim()
                };

                controller.enqueue(
                  encoder.encode(`data: ${JSON.stringify({
                    type: 'concepts',
                    extractedConcepts: extractedConcept,
                    progress: { processed: processedResponses, total: totalResponses }
                  })}\n\n`)
                );
              }
            }
          }
        }

        const allConceptFrequencies = new Map<string, number>();
        allConcepts.forEach(concept => {
          allConceptFrequencies.set(concept, (allConceptFrequencies.get(concept) || 0) + 1);
        });

        const clusters = await runConceptClustering(allConceptFrequencies, subgroupConcepts, userApiKeys);
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify({ type: 'clusters', clusters })}\n\n`)
        );
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify({
            type: 'complete',
            message: 'Concept extraction completed',
            progress: { processed: totalResponses, total: totalResponses }
          })}\n\n`)
        );
        controller.close();
      } catch (error) {
        console.error('Concept extraction failed:', error);
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify({
            type: 'error',
            error: error instanceof Error ? error.message : 'Unknown error'
          })}\n\n`)
        );
        controller.close();
      }
    },
    cancel() {
      console.log('Stream cancelled by client');
    }
  });

  return new NextResponse(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
}

function getDemographics(demos?: string[]): { category: string; value: string }[] {
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

async function extractConcepts(
  text: string,
  userApiKeys: Record<'openai' | 'anthropic' | 'huggingface'|'deepseek'|'mistral', string>
) {
  if (!userApiKeys?.openai) throw new Error("Missing OpenAI API key.");

  const openai = new OpenAI({ apiKey: userApiKeys.openai });
  const sanitizedText = text.replace(/[\n\r]+/g, ' ').replace(/"/g, '\\"').trim();

  const prompt = `Extract key concepts and themes from the following response text.
Return ONLY a single line containing a list in this format: [concept1, concept2, concept3].
Instructions:
- Extract all actionable or detailed concepts mentioned.
Example:
For the response text: "I'm sorry to hear that you're feeling anxious. Here are some tips that may help you manage your anxiety: 1. Take deep breaths, 2. Practice mindfulness, 3. Exercise, 4. Talk to someone, 5. Get enough sleep, 6. Limit caffeine and alcohol, 7. Practice self-care..."
The output should be: [deep breaths, practice mindfulness, exercise, talk to someone, proper sleep, limit substances, self-care]
Response text: "${sanitizedText}"
Return ONLY the list in [item1, item2, item3] format, with clear, concise concepts in plain English.`;

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: "You are a concept extraction system. Return only a single line containing a bracketed list of concepts. Format: [concept1, concept2, concept3]."
        },
        { role: "user", content: prompt }
      ],
      temperature: 0.3,
    });

    const content = completion.choices[0].message.content || '';
    const match = content.match(/\[(.*?)\]/);
    if (match) {
      return match[1].split(',').map(concept => concept.trim()).filter(concept => concept && concept !== '...');
    }
    console.log("No concepts found in response");
    return [];
  } catch (error) {
    console.error('OpenAI API error:', error);
    throw error;
  }
}
