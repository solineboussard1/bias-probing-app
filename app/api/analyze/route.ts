import { NextRequest, NextResponse } from 'next/server';
import { runAnalysisPipeline } from '@/app/lib/pipeline-service';
import { SelectedParams } from '@/app/types/pipeline';

export async function POST(request: NextRequest): Promise<Response> {
  try {
    // Parse request body
    const params = await request.json() as SelectedParams & {
      userApiKeys: Record<'openai' | 'anthropic' | 'huggingface'| 'deepseek'| 'mistral', string> & Record<string, string>;
    };
    console.log('API POST: Received parameters:', params);


    // Adjust payload based on domain type
    const isCustomDomain = params.domain === 'custom';
    const processedParams = {
      ...params,
      primaryIssues: isCustomDomain ? [] : params.primaryIssues,
      recommendations: isCustomDomain ? [] : params.recommendations,
      relevantStatements: isCustomDomain ? [] : params.relevantStatements,
      context: isCustomDomain ? 'Custom' : params.context,
    };
    console.log('API POST: Processed parameters:', processedParams);


    // Validate required fields
    if (!processedParams.model) {
      throw new Error('Missing required parameter: model');
    }
    if (!processedParams.iterations || processedParams.iterations < 1) {
      throw new Error('Invalid iterations value: must be at least 1');
    }
    if (!isCustomDomain && (!processedParams.primaryIssues || processedParams.primaryIssues.length === 0)) {
      throw new Error('Missing required parameter: primaryIssues (not applicable for custom domain)');
    }

    // Stream for server-sent events
    const stream = new TransformStream();
    const writer = stream.writable.getWriter();
    const encoder = new TextEncoder();

    const response = new NextResponse(stream.readable, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive',
        'Content-Encoding': 'none', // Disable compression to help with streaming
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization'
      },
    });
    
    

    type ProgressUpdate = {
      type: string;
      message?: string;
    };

    // Run the analysis pipeline with proper parameters
    console.log('API POST: Starting analysis pipeline');

    runAnalysisPipeline(processedParams, params.userApiKeys, async (update: ProgressUpdate) => {
      console.log('Sending SSE update:', update);
      await writer.write(encoder.encode(`data: ${JSON.stringify(update)}\n\n`));
    })
      .then(async (result) => {
        await writer.write(encoder.encode(`data: ${JSON.stringify({ type: 'complete', result })}\n\n`));
        await writer.close();
      })
      .catch(async (error) => {
        console.error('Pipeline error:', error);
        await writer.write(encoder.encode(`data: ${JSON.stringify({ type: 'error', error: error.message })}\n\n`));
        await writer.close();
      });

    return response;
  } catch (error) {
    console.error('Analysis pipeline error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Unknown error occurred' },
      { status: 500 }
    );
  }
}
