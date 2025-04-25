'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Minimize2, BarChart3, Eye, EyeOff, Save, Slice, Download, ChevronDown, ChevronUp, Upload, } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue, } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { toast } from 'sonner';
import { TooltipProvider, Tooltip, TooltipTrigger, TooltipContent, } from "@/components/ui/tooltip";
import { useHotkeys } from 'react-hotkeys-hook';
import { AnalysisResult, ClusterOutput, SelectedParams, PromptResult, LDAResult, AgreementScores, ExtractedConcepts, AllResults } from './types/pipeline';
import { ConceptVisualizations } from '@/components/ui/ConceptVisualizations';
import { LDAVisualizations } from '@/components/ui/LDAVisualizations';
import { Input } from "@/components/ui/input";
import { EmbeddingsVisualizations } from "@/components/ui/EmbeddingsVisualizations";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AgreementScoreVisualizations } from "@/components/ui/AgreementScoreVisualizations";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger, } from "@/components/ui/dropdown-menu";
import { downloadCSV, createMergedAnalysisCSV } from "@/app/lib/csv-utils";
import { PipelineParams, PaginationState, ExtractionProgress, SavedAnalysis, DEFAULT_PIPELINE_PARAMS } from "@/app/lib/constants";

const ITEMS_PER_PAGE = 8;

const API_KEYS = [
  { provider: "openai" },
  { provider: "anthropic" },
  { provider: "huggingface" },
  { provider: "deepseek" },
  { provider: "mistral" }
];

export default function Home() {
  const customRelevantStatementsRef = useRef<string[]>([]);
  const [pipelineParams] = useState<PipelineParams>(DEFAULT_PIPELINE_PARAMS);
  const [selectedParams, setSelectedParams] = useState<SelectedParams>({
    model: DEFAULT_PIPELINE_PARAMS.models[0],
    domain: 'healthcare',
    primaryIssues: [DEFAULT_PIPELINE_PARAMS.domainPatterns.healthcare.primaryIssues[0]],
    recommendations: [...DEFAULT_PIPELINE_PARAMS.domainPatterns.healthcare.recommendationPatterns],
    irrelevantStatements: [...DEFAULT_PIPELINE_PARAMS.irrelevantStatements],
    relevantStatements: [...DEFAULT_PIPELINE_PARAMS.relevantStatements.healthcare],
    templates: [...DEFAULT_PIPELINE_PARAMS.domainPatterns.healthcare.baselineTemplates],
    perspectives: [...DEFAULT_PIPELINE_PARAMS.perspectives],
    demographics: {
      genders: [DEFAULT_PIPELINE_PARAMS.demographics.genders[0]],
      ages: [DEFAULT_PIPELINE_PARAMS.demographics.ages[0]],
      ethnicities: [DEFAULT_PIPELINE_PARAMS.demographics.ethnicities[0]],
      socioeconomic: [DEFAULT_PIPELINE_PARAMS.demographics.socioeconomic[0]]
    },
    context: DEFAULT_PIPELINE_PARAMS.contexts[0],
    relevanceOptions: [DEFAULT_PIPELINE_PARAMS.relevanceOptions[0]],
    questionTypes: [DEFAULT_PIPELINE_PARAMS.questionTypes[0]],
    iterations: 1
  });

  // State management
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [savedAnalyses, setSavedAnalyses] = useState<SavedAnalysis[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [pagination, setPagination] = useState<PaginationState>({});
  const [progress, setProgress] = useState<{
    currentPrompt?: string;
    iteration?: number;
    totalPrompts?: number;
    completedPrompts?: number;
    message?: string;
  } | null>(null);
  const [showKeys, setShowKeys] = useState(false);
  const [apiKeys, setApiKeys] = useState(
    API_KEYS.map((p) => ({ provider: p.provider, key: "" }))
  );

  const updateApiKey = (provider: string, newKey: string) => {
    setApiKeys((prev) =>
      prev.map((api) =>
        api.provider === provider ? { ...api, key: newKey } : api
      )
    );
  };

  // Add new state for section visibility
  const [configSectionsExpanded, setConfigSectionsExpanded] = useState({
    modelConfig: true,
    parameters: true
  });

  // Add state for concept distributions
  const [conceptData, setConceptData] = useState<{
    concepts: Map<string, number>;
    demographicDistributions: Map<string, Map<string, Map<string, number>>>;
    clusters?: ClusterOutput;
    rawResults?: AnalysisResult[];
    extractedConcepts?: ExtractedConcepts[];
  }>({
    concepts: new Map(),
    demographicDistributions: new Map()
  });

  // Add new state for file upload
  const [isProcessingFile, setIsProcessingFile] = useState(false);

  const [isLoadingConceptsFile, setIsLoadingConceptsFile] = useState(false);

  // Update extraction progress state to handle multiple types
  const [extractionProgress, setExtractionProgress] = useState<{
    llm?: ExtractionProgress;
    lda?: ExtractionProgress;
    embeddings?: ExtractionProgress;
  }>({});

  // Add extraction loading state
  const [isExtracting, setIsExtracting] = useState<{
    llm: boolean;
    lda: boolean;
    embeddings: boolean;
  }>({ llm: false, lda: false, embeddings: false });

  // Add new state for LDA results
  const [ldaResults, setLdaResults] = useState<LDAResult | null>(null);

  // Add this state for embeddings results
  const [embeddingsResults, setEmbeddingsResults] = useState<{
    cluster_id: number;
    representative_responses: string[];
    coordinates: number[][];
    embeddings: number[][];
    size: number;
    distribution: { [key: string]: number };
  }[]>([]);

  // Add new state
  const [agreementData, setAgreementData] = useState<AgreementScores | null>(null);

  // Mobile responsiveness
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 640);
      if (window.innerWidth < 640) {
        setIsSidebarOpen(false);
      }
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  // Add hotkey handler
  useHotkeys('meta+e, ctrl+e', (e) => {
    e.preventDefault();
    setIsSidebarOpen(prev => !prev);
  }, {
    enableOnFormTags: true,
    preventDefault: true
  });

  // Add this useEffect to collapse sections when analysis starts or results load
  useEffect(() => {
    if (isAnalyzing || progress || analysisResults.length > 0) {
      setConfigSectionsExpanded({
        modelConfig: false,
        parameters: false
      });
    }
  }, [isAnalyzing, progress, analysisResults.length]);

  const handleAnalyze = async () => {
    // Basic validation of required fields
    console.log('handleAnalyze: Starting analysis');

    if (!selectedParams.model || (selectedParams.domain !== 'custom' && selectedParams.primaryIssues.length === 0)) {
      toast.error('Please select a model and at least one primary issue');
      return;
    }

    const userApiKeys = apiKeys.reduce((acc, { provider, key }) => {
      acc[provider as 'openai' | 'anthropic' | 'huggingface' | 'deepseek' | 'mistral'] = key;
      return acc;
    }, {} as Record<'openai' | 'anthropic' | 'huggingface' | 'deepseek' | 'mistral', string>);

    console.log('handleAnalyze: User API keys provided for providers:',
      apiKeys.map((a) => ({ provider: a.provider, keyExists: !!a.key })));

    // Map models to their providers
    const modelProviderMap: Record<string, 'openai' | 'anthropic' | 'huggingface' | 'deepseek' | 'mistral'> = {
      'gpt-4o': 'openai',
      'gpt-4o-mini': 'openai',
      'gpt-o1-mini': 'openai',
      'claude-3-5-sonnet-latest': 'anthropic',
      'mistral-7b': 'mistral',
      'llama-3-8b': 'huggingface',
      'deepseek-chat': 'deepseek'
    };

    const provider = modelProviderMap[selectedParams.model];
    if (!provider) {
      toast.error('Selected model is not supported.');
      return;
    }

    if (!userApiKeys[provider]) {
      toast.error(`Please provide your ${provider} API key`);
      return;
    }

    const payload = {
      model: selectedParams.model,
      domain: selectedParams.domain,
      primaryIssues: selectedParams.domain === 'custom' ? [] : selectedParams.primaryIssues,
      recommendations: selectedParams.domain === 'custom' ? [] : selectedParams.recommendations,
      relevantStatements: selectedParams.relevantStatements,
      irrelevantStatements: selectedParams.irrelevantStatements,
      templates: selectedParams.domain === 'custom' ? [] : selectedParams.templates,
      perspectives: selectedParams.perspectives,
      demographics: selectedParams.demographics,
      context: selectedParams.domain === 'custom' ? 'Custom' : selectedParams.context,
      relevanceOptions: selectedParams.relevanceOptions,
      questionTypes: selectedParams.domain === 'custom' ? ['Open-Ended'] : selectedParams.questionTypes,
      iterations: selectedParams.iterations,
      customPrompts: selectedParams.domain === 'custom' ? (selectedParams.customPrompts || []) : [],
      userApiKeys: userApiKeys,
    };

    console.log('handleAnalyze: Payload being sent:', payload);


    setIsAnalyzing(true);
    setProgress(null);
    setConceptData({ concepts: new Map(), demographicDistributions: new Map() });
    setLdaResults(null);
    setEmbeddingsResults([]);
    setAgreementData(null);
    setAnalysisResults([]);

    try {
      const getApiBase = () => {
        if (window.location.hostname === 'localhost') {
          return '/api';
        } else {
          return 'https://bias-probing.onrender.com/api';
        }
      };

      const response = await fetch(`${getApiBase()}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      console.log('handleAnalyze: API response status:', response.status);

      if (!response.ok) throw new Error('Analysis request failed');

      const reader = response.body?.getReader();
      if (!reader) throw new Error('Failed to get response reader');

      const decoder = new TextDecoder();
      const results: AnalysisResult[] = [];
      let buffer = ''; // Add buffer for incomplete chunks

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          console.log('Reader done');
          break;
        }
        const decoded = decoder.decode(value, { stream: true });
        console.log('Raw chunk received:', decoded);
        buffer += decoded;
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (line.trim().startsWith('data: ')) {
            console.log('Processing line:', line);
            try {
              const data = JSON.parse(line.slice(5));
              console.log('Parsed SSE data:', data);
              if (data.type === 'complete') {
                results.push(data.result);
                setAnalysisResults(prev => [...prev, data.result]);
                toast.success('Analysis completed successfully');
              } else if (data.type === 'error') {
                throw new Error(data.error);
              } else {
                setProgress({
                  currentPrompt: data.prompt,
                  iteration: data.iteration,
                  totalPrompts: data.totalPrompts,
                  completedPrompts: data.completedPrompts,
                  message: data.message
                });
                console.log('Progress update:', data);
              }
            } catch (parseError) {
              console.error('Failed to parse SSE data chunk:', parseError);
              console.log('Problematic line:', line);
            }
          } else {
            console.log('Skipping non-data line:', line);
          }
        }
      }


      // Process any remaining data in buffer
      if (buffer.trim()) {
        console.log('Final buffer contents:', buffer);
        try {
          const data = JSON.parse(buffer.slice(5));
          console.log('Final parsed SSE data:', data);
          if (data.type === 'complete') {
            results.push(data.result);
            setAnalysisResults(prev => [...prev, data.result]);
          }
        } catch (parseError) {
          console.error('Failed to parse final buffer:', parseError);
        }
      }


      // Only proceed with concept extraction if we have results
      console.log('Final analysis results:', results);
      if (results.length > 0) {
        await extractConcepts(results);
      } else {
        throw new Error('No valid results received from analysis');
      }

    } catch (error) {
      console.error('Pipeline failed:', error);
      toast.error('Pipeline failed. Please try again.', {
        description: error instanceof Error ? error.message : 'Unknown error occurred'
      });
    } finally {
      setIsAnalyzing(false);
      setProgress(null);
    }
  };


  const saveAnalysis = () => {
    if (analysisResults.length === 0) return;

    // Create the analysis object
    const newAnalysis: SavedAnalysis = {
      id: crypto.randomUUID(),
      name: `Analysis ${savedAnalyses.length + 1}`,
      results: analysisResults,
      created_at: new Date().toISOString()
    };

    // Save to local storage
    setSavedAnalyses(prev => [...prev, newAnalysis]);
    localStorage.setItem('savedAnalyses', JSON.stringify([...savedAnalyses, newAnalysis]));

    // Download the JSON file
    const dataStr = JSON.stringify(newAnalysis, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `spicex-analysis-${newAnalysis.id}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    toast.success('Analysis saved and downloaded successfully');
  };

  const handlePromptUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const content = e.target?.result as string;
        const parsed = JSON.parse(content);
        let prompts: string[] = [];
        let uploadedRelevantStatements: string[] = [];
        // Check for object format, only accepting prompts and relevantStatements from the file
        if (typeof parsed === 'object' && !Array.isArray(parsed)) {
          prompts = parsed.prompts || [];
          uploadedRelevantStatements = parsed.relevantStatements || [];
        } else if (Array.isArray(parsed)) {
          prompts = parsed;
        } else {
          throw new Error('Invalid JSON format for prompts.');
        }

        // For custom domain, keep the full list in a ref
        customRelevantStatementsRef.current = uploadedRelevantStatements;

        setSelectedParams(prev => ({
          ...prev,
          customPrompts: prompts,
          // For toggling, initialize with all uploaded statements as active.
          relevantStatements: uploadedRelevantStatements.length > 0
            ? uploadedRelevantStatements
            : prev.relevantStatements,
        }));

        console.log('Custom prompts loaded:', prompts);
      } catch (error) {
        console.error('Error processing file:', error);
      }
    };
    reader.onerror = (error) => {
      console.error('File reading error:', error);
    };
    reader.readAsText(file);
  };

  const handleFileUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsProcessingFile(true);
    // toast.info('Processing uploaded results file...');

    try {
      // Read the file
      const text = await file.text();
      const uploadedResults = JSON.parse(text);

      // Check if it's a single analysis or array of analyses
      const results = uploadedResults.results ||
        (Array.isArray(uploadedResults) ? uploadedResults : [uploadedResults]);

      if (!Array.isArray(results) || !results.length) {
        throw new Error('Invalid results format');
      }

      // Set the results
      setAnalysisResults(results);
      toast.success('Results loaded successfully');

      // Extract concepts from the loaded results
      await extractConcepts(results);

    } catch (error) {
      console.error('File processing failed:', error);
      toast.error('Failed to process results file', {
        description: error instanceof Error ? error.message : 'Invalid file format'
      });
    } finally {
      setIsProcessingFile(false);
      if (event.target) {
        event.target.value = ''; // Reset file input
      }
    }
  }, []);

  const extractConcepts = async (results: AnalysisResult[]) => {
    let llmAbortController: AbortController | null = null;
    let ldaAbortController: AbortController | null = null;

    try {
      setIsExtracting({ llm: true, lda: true, embeddings: true });

      const userApiKeys = apiKeys.reduce((acc, { provider, key }) => {
        acc[provider as 'openai' | 'anthropic' | 'huggingface' | 'deepseek' || 'mistral'] = key;
        return acc;
      }, {} as Record<'openai' | 'anthropic' | 'huggingface' | 'deepseek' | 'mistral', string>);

      // Validate API key presence
      if (!userApiKeys.openai && !userApiKeys.huggingface) {
        toast.error('Please provide at least HuggingFace and OpenAI API key.');
        return;
      }

      // Track all extracted concepts for later use
      const allExtractedConcepts: ExtractedConcepts[] = [];

      // Start both extractions in parallel
      llmAbortController = new AbortController();
      ldaAbortController = new AbortController();

      const getApiBase = () => {
        if (window.location.hostname === 'localhost') {
          return '/api';
        } else {
          return 'https://bias-probing.onrender.com/api';
        }
      };

      // Using the helper for llm extraction
      const llmPromise = fetch(`${getApiBase()}/llm-extract-concepts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ results, userApiKeys }),
        signal: llmAbortController.signal,
      });

      // And for LDA extraction
      const ldaPromise = fetch(`${getApiBase()}/lda-extract-concepts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(results),
        signal: ldaAbortController.signal,
      });

      const [llmResponse, ldaResponse] = await Promise.all([llmPromise, ldaPromise]);

      if (!llmResponse.ok || !ldaResponse.ok) {
        throw new Error('One or more extraction methods failed');
      }

      const decoder = new TextDecoder();

      // Update LLM progress handling
      const llmReader = llmResponse.body?.getReader();
      if (llmReader) {
        try {
          while (true) {
            const { done, value } = await llmReader.read();
            if (done) break;
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(5));
                switch (data.type) {
                  case 'extraction_progress':
                    setExtractionProgress(prev => ({
                      ...prev,
                      llm: {
                        processed: data.progress.processed,
                        total: data.progress.total,
                        message: data.message,
                        type: 'llm',
                      },
                    }));
                    break;
                  case 'concepts': {
                    const extractedConcepts = data.extractedConcepts as ExtractedConcepts;
                    allExtractedConcepts.push(extractedConcepts);

                    setConceptData(prev => {
                      const newConcepts = new Map(prev.concepts);
                      const newDemographicDistributions = new Map(prev.demographicDistributions);

                      const expectedDemographics = ['genders', 'ethnicities', 'ages', 'socioeconomic'];

                      extractedConcepts.concepts.forEach((concept: string) => {
                        newConcepts.set(concept, (newConcepts.get(concept) || 0) + 1);

                        expectedDemographics.forEach((demoCategory) => {
                          let demoValue: string | undefined;

                          if (extractedConcepts.demographics && extractedConcepts.demographics.length > 0) {
                            const match = extractedConcepts.demographics.find(demo => demo.category === demoCategory);
                            if (match) {
                              demoValue = match.value;
                            }
                          } else {
                            demoValue = 'baseline';
                          }

                          if (demoValue) {
                            if (!newDemographicDistributions.has(demoCategory)) {
                              newDemographicDistributions.set(demoCategory, new Map());
                            }
                            const categoryMap = newDemographicDistributions.get(demoCategory)!;

                            if (!categoryMap.has(demoValue)) {
                              categoryMap.set(demoValue, new Map());
                            }
                            const valueMap = categoryMap.get(demoValue)!;

                            valueMap.set(concept, (valueMap.get(concept) || 0) + 1);
                          }
                        });
                      });

                      return {
                        ...prev,
                        concepts: newConcepts,
                        demographicDistributions: newDemographicDistributions,
                        rawResults: results,
                        extractedConcepts: allExtractedConcepts,
                      };
                    });

                    break;
                  }

                  case 'clusters': {
                    const clustersData = { ...data.clusters };
                    clustersData.all = Array.isArray(clustersData.all)
                      ? clustersData.all
                      : Object.values(clustersData.all || {});

                    setConceptData(prev => ({
                      ...prev,
                      clusters: clustersData,
                      rawResults: results,
                      extractedConcepts: allExtractedConcepts,
                    }));
                    break;
                  }
                }
              }
            }
          }
        } finally {
          llmReader.releaseLock();
        }
      }

      // Update LDA progress handling
      const ldaReader = ldaResponse.body?.getReader();
      if (ldaReader) {
        try {
          while (true) {
            const { done, value } = await ldaReader.read();
            if (done) break;
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(5));
                switch (data.type) {
                  case 'extraction_progress':
                    setExtractionProgress(prev => ({
                      ...prev,
                      lda: {
                        processed: data.progress.processed,
                        total: data.progress.total,
                        message: data.message,
                        type: 'lda',
                      },
                    }));
                    break;
                  case 'lda_concepts':
                    setLdaResults({
                      topics: data.topics,
                      distributions: data.distributions,
                      demographicDistributions: data.demographicDistributions,
                    });
                    break;
                  case 'complete':
                    setIsExtracting(prev => ({ ...prev, lda: false }));
                    setExtractionProgress(prev => ({ ...prev, lda: undefined }));
                    break;
                  case 'error':
                    toast.error(`LDA Error: ${data.error}`);
                    setIsExtracting(prev => ({ ...prev, lda: false }));
                    break;
                }
              }
            }
          }
        } finally {
          ldaReader.releaseLock();
        }
      }

      // Add embeddings extraction
      await extractEmbeddings(results);
    } catch (error) {
      console.error('Concept extraction failed:', error);
      toast.error('Concept extraction failed', {
        description: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      setIsExtracting({ llm: false, lda: false, embeddings: false });
      setExtractionProgress({});
      if (llmAbortController) llmAbortController.abort();
      if (ldaAbortController) ldaAbortController.abort();
    }
  };

  // Update the extractEmbeddings function
  const extractEmbeddings = async (results: AnalysisResult[]) => {
    try {
      setIsExtracting(prev => ({ ...prev, embeddings: true }));
      const userApiKeys = apiKeys.reduce((acc, { provider, key }) => {

        acc[provider as 'openai' | 'anthropic' | 'huggingface' | 'deepseek' | 'mistral'] = key;

        return acc;
      }, {} as Record<'openai' | 'anthropic' | 'huggingface' | 'deepseek' | 'mistral', string>);

      // Validate API key presence
      if (!userApiKeys.openai && !userApiKeys.anthropic && !userApiKeys.huggingface) {
        toast.error('Please provide at least one API key.');
        return;
      }
      const getApiBase = () => {
        if (window.location.hostname === 'localhost') {
          return '/api';
        } else {
          return 'https://bias-probing.onrender.com/api';
        }
      };

      const response = await fetch(`${getApiBase()}/embeddings-extract-concepts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ results, userApiKeys })
      });

      if (!response.ok) {
        throw new Error('Embeddings extraction failed');
      }

      const data = await response.json();
      if (data.error) {
        throw new Error(data.error);
      }

      setEmbeddingsResults(data);
      setIsExtracting(prev => ({ ...prev, embeddings: false }));

    } catch (error) {
      console.error('Embeddings extraction failed:', error);
      toast.error('Failed to extract embeddings');
      setIsExtracting(prev => ({ ...prev, embeddings: false }));
    }
  };

  const fileInputRef = useRef<HTMLInputElement>(null);

  const conceptsFileInputRef = useRef<HTMLInputElement>(null);


  const handleConceptsUploadClick = () => {
    conceptsFileInputRef.current?.click();
  };

  // Update the ExtractionProgressDisplay component
  const ExtractionProgressDisplay = () => {
    return (
      <>
        {(isExtracting.llm || isExtracting.lda ||
          extractionProgress.llm || extractionProgress.lda) && (
            <div className="space-y-4 mb-4">
              {/* LLM Progress */}
              {(isExtracting.llm || extractionProgress.llm) && (
                <div className="space-y-2 p-4 bg-muted rounded-lg border">
                  <div className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">
                        {extractionProgress.llm?.message || 'Initializing LLM concept extraction...'}
                      </span>
                      {extractionProgress.llm && (
                        <span className="font-medium">
                          {extractionProgress.llm.processed}/{extractionProgress.llm.total} responses
                        </span>
                      )}
                    </div>
                    <div className="w-full bg-secondary rounded-full h-2">
                      <div
                        className="bg-primary rounded-full h-2 transition-all duration-300"
                        style={{
                          width: extractionProgress.llm
                            ? `${(extractionProgress.llm.processed / extractionProgress.llm.total) * 100}%`
                            : '0%'
                        }}
                      />
                    </div>
                  </div>
                </div>
              )}

            </div>
          )}
      </>
    );
  };

  // Update the calculateAgreementScores function
  const calculateAgreementScores = async () => {
    try {
      if (!conceptData.extractedConcepts || !ldaResults || !embeddingsResults.length) {
        toast.error('Missing required data for agreement calculation');
        return;
      }

      // Create merged CSV data
      const mergedData = createMergedAnalysisCSV(
        analysisResults,
        conceptData.extractedConcepts,
        ldaResults,
        embeddingsResults,
        conceptData.clusters?.all || []
      );

      const getApiBase = () => {
        if (window.location.hostname === 'localhost') {
          return '/api';
        } else {
          return 'https://bias-probing.onrender.com/api';
        }
      };

      const response = await fetch(`${getApiBase()}/calculate-agreement`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mergedCsv: mergedData })
      });



      if (!response.ok) {
        const errorMessage = await response.text();
        console.error('Error response:', errorMessage);
        throw new Error('Failed to calculate agreement scores');
      }

      const data = await response.json();
      setAgreementData(data);
    } catch (error) {
      console.error('Error calculating agreement scores:', error);
      toast.error('Failed to calculate agreement scores');
    }
  };

  // Call calculateAgreementScores when all three extraction methods are complete
  const [isDataReady, setIsDataReady] = useState(false);

  useEffect(() => {
    const allDataPresent =
      conceptData.concepts.size > 0 &&
      ldaResults !== null &&
      embeddingsResults.length > 0;

    if (allDataPresent) {
      setIsDataReady(true);
    }
  }, [conceptData.concepts.size, ldaResults, embeddingsResults.length]);

  useEffect(() => {
    if (isDataReady) {
      calculateAgreementScores();
      setIsDataReady(false);
    }
  }, [isDataReady]);


  const handleDownloadResults = () => {
    try {
      // Prepare the data
      const allResults: AllResults = {
        analysisResults,
        conceptResults: {
          llm: {
            concepts: Array.from(conceptData.concepts.entries()),

            demographicDistributions: Array.from(conceptData.demographicDistributions.entries()).map(
              ([demo, categoryMap]) => [
                demo,
                new Map(
                  Array.from(categoryMap.entries()).flatMap(([, valueMap]) =>
                    Array.from(valueMap.entries())
                  )
                )
              ]
            ),

            extractedConcepts: conceptData.extractedConcepts || [],
            clusters: conceptData.clusters
          },
          lda: ldaResults ? {
            topics: ldaResults.topics,
            distributions: ldaResults.distributions,
            demographicDistributions: ldaResults.demographicDistributions,
            error: ldaResults.error,
          } : null,
          embeddings: embeddingsResults
        }
      };

      // Convert Maps to JSON-safe objects
      const serializedResults = {
        ...allResults,
        conceptResults: {
          ...allResults.conceptResults,
          llm: {
            ...allResults.conceptResults.llm,
            demographicDistributions: allResults.conceptResults.llm.demographicDistributions.map(
              ([demo, conceptMap]) => [
                demo,
                Object.fromEntries(conceptMap) 
              ]
            )
          }
        }
      };

      // Create and trigger download
      const blob = new Blob([JSON.stringify(serializedResults, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `concept-analysis-results-${new Date().toISOString()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      toast.success('Results downloaded successfully');
    } catch (error) {
      console.error('Error downloading results:', error);
      toast.error('Failed to download results');
    }
  };

  const handleUploadResults = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsLoadingConceptsFile(true);
    toast.info('Processing uploaded concepts file...');

    try {
      const text = await file.text();
      const allResults: AllResults = JSON.parse(text);

      // Populate all the states
      setAnalysisResults(allResults.analysisResults);

      // Reconstruct the Maps for concept distributions as a three-level Map.
      const demographicDistributionsMap = new Map(
        allResults.conceptResults.llm.demographicDistributions.map(
          ([demo, categoryObj]) => [
            demo,
            new Map(
              Object.entries(categoryObj).map(
                ([demoValue, valueObj]) => [
                  demoValue,
                  new Map<string, number>(Object.entries(valueObj) as [string, number][])
                ]
              )
            )
          ]
        )
      );


      // Reconstruct the concepts Map
      const conceptsMap = new Map(allResults.conceptResults.llm.concepts);

      setConceptData({
        concepts: conceptsMap,
        demographicDistributions: demographicDistributionsMap,
        clusters: allResults.conceptResults.llm.clusters
          ? {
            all: allResults.conceptResults.llm.clusters.all,
            demographics: allResults.conceptResults.llm.clusters.demographics
          }
          : undefined,
        rawResults: allResults.analysisResults,
        extractedConcepts: allResults.conceptResults.llm.extractedConcepts,
      });

      setLdaResults(allResults.conceptResults.lda);
      setEmbeddingsResults(allResults.conceptResults.embeddings);
      calculateAgreementScores();

    } catch (error) {
      console.error('File processing failed:', error);
      toast.error('Failed to process results file', {
        description: error instanceof Error ? error.message : 'Invalid file format',
      });
    } finally {
      setIsLoadingConceptsFile(false);
      if (event.target) {
        event.target.value = '';
      }
    }
  }, []);

  return (
    <div className="flex h-[100dvh] overflow-hidden bg-background">
      {/* Sidebar */}
      <div
        className={`hidden sm:block ${isSidebarOpen ? 'sm:w-64' : 'sm:w-12'} 
          border-r border-border bg-muted/50 transition-all duration-300 ease-in-out overflow-hidden sticky top-0 h-screen`}
      >
        {isSidebarOpen ? (
          <div className="flex flex-col h-full">
            <div className="p-3 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Slice className="h-4 w-4" />
                  <h2 className="font-semibold tracking-tight">LLM Bias Probing</h2>
                </div>
                <div className="flex items-center gap-1">
                  <kbd className="text-[10px] font-mono px-1 py-0.5 border rounded bg-muted">⌘E</kbd>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7"
                    onClick={() => setIsSidebarOpen(false)}
                  >
                    <Minimize2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>
            {/* API Keys Section */}
            <div className="p-4 border-b border-border">
              <div className="flex flex-row gap-2 items-center">
                <p className="font-bold">Add API keys</p>
                <TooltipProvider delayDuration={0}>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => setShowKeys(!showKeys)}
                        className="h-full/2 w-full/2"
                      >
                        {showKeys ? (
                          <Eye className="h-4 w-4" />
                        ) : (
                          <EyeOff className="h-4 w-4" />
                        )}
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>{showKeys ? "Hide API keys" : "Show API keys"}</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              {apiKeys.map((apiKey) => (
                <div
                  className="grid w-full items-center gap-1.5 mt-2"
                  key={`api-key-provider-${apiKey.provider}`}
                >
                  <Label className="capitalize">{apiKey.provider}</Label>
                  <Input
                    type={showKeys ? "text" : "password"}
                    value={apiKey.key}
                    onChange={(e) => updateApiKey(apiKey.provider, e.target.value)}
                  />
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="w-10 flex flex-col items-center py-2 gap-2 opacity-100 transition-opacity duration-300 ease-in-out">
            <TooltipProvider delayDuration={0}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => setIsSidebarOpen(true)}
                  >
                    <Slice className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="right">
                  Expand Sidebar (⌘E)
                </TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon" className="h-8 w-8">
                    <BarChart3 className="h-4 w-4 text-muted-foreground" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="right">
                  View Saved Analyses
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        )}
      </div>

      {/* Mobile Sidebar Overlay */}
      {isSidebarOpen && isMobile && (
        <div
          className="fixed inset-0 bg-background/80 backdrop-blur-sm z-40 sm:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Main Content */}
      <div className="flex-1 overflow-auto">
        {/* Conditional API Key Banner */}
        {apiKeys.every(api => !api.key) && (
          <div className="bg-yellow-100 border border-yellow-300 text-yellow-800 p-4 rounded-lg text-center m-4">
            <p>Please add your API key to get started.</p>
            <p>
              Tip: Press <kbd className="font-mono bg-gray-200 p-1 rounded">⌘E</kbd> or click the sidebar button to reveal the API keys panel.
            </p>
          </div>
        )}

        <div className={`
          ${(conceptData.concepts.size > 0 || ldaResults || embeddingsResults.length > 0)
            ? 'flex flex-col sm:flex-row gap-6 h-[100dvh] overflow-hidden'
            : 'flex justify-center p-6 min-h-[100dvh]'}
        `}>
          <div className={`
            ${(conceptData.concepts.size > 0 || ldaResults || embeddingsResults.length > 0)
              ? 'w-full sm:flex-[0.8] p-4 sm:p-6 overflow-auto'
              : 'w-full max-w-4xl'}
          `}>
            {/* Configuration section */}
            <div className="space-y-4">
              <div className="space-y-3">
                <div
                  className="border-b pb-1 flex justify-between items-center cursor-pointer"
                  onClick={() => setConfigSectionsExpanded(prev => ({
                    ...prev,
                    modelConfig: !prev.modelConfig
                  }))}
                >
                  <h3 className="text-lg font-semibold tracking-tight">Model Configuration</h3>
                  {configSectionsExpanded.modelConfig ? (
                    <ChevronUp className="h-4 w-4" />
                  ) : (
                    <ChevronDown className="h-4 w-4" />
                  )}
                </div>

                {configSectionsExpanded.modelConfig && (
                  <div className="space-y-2 p-3 bg-muted/50 rounded-lg border">
                    <div className="space-y-1">
                      <Label className="text-sm font-medium">Model</Label>
                      <Select
                        value={selectedParams.model}
                        onValueChange={(value) => setSelectedParams(prev => ({
                          ...prev,
                          model: value
                        }))}
                      >
                        <SelectTrigger className="w-full">
                          <SelectValue placeholder="Choose an LLM" />
                        </SelectTrigger>
                        <SelectContent>
                          {pipelineParams.models.map(model => (
                            <SelectItem key={model} value={model}>
                              {model}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                )}
              </div>

              {/* Parameters Section */}
              <div className="space-y-3">
                <div
                  className="border-b pb-1 flex justify-between items-center cursor-pointer"
                  onClick={() =>
                    setConfigSectionsExpanded(prev => ({
                      ...prev,
                      parameters: !prev.parameters,
                    }))
                  }
                >
                  <h3 className="text-lg font-semibold tracking-tight">Parameters</h3>
                  {configSectionsExpanded.parameters ? (
                    <ChevronUp className="h-4 w-4" />
                  ) : (
                    <ChevronDown className="h-4 w-4" />
                  )}
                </div>

                {configSectionsExpanded.parameters && (
                  <div className="space-y-4 p-3 bg-muted/50 rounded-lg border">
                    {/* Domain Selection */}
                    <div className="space-y-1">
                      <Label>Domain</Label>
                      <Select
                        value={selectedParams.domain}
                        onValueChange={(value) => {
                          if (value === 'custom') {
                            setSelectedParams(prev => ({
                              ...prev,
                              domain: value,
                              primaryIssues: [],
                              recommendations: [],
                              relevantStatements: prev.relevantStatements,
                              templates: [],
                              relevanceOptions: ["Neutral"],
                            }));
                          } else {
                            setSelectedParams(prev => ({
                              ...prev,
                              domain: value,
                              primaryIssues: [DEFAULT_PIPELINE_PARAMS.domainPatterns[value].primaryIssues[0]],
                              recommendations: [...DEFAULT_PIPELINE_PARAMS.domainPatterns[value].recommendationPatterns],
                              relevantStatements: DEFAULT_PIPELINE_PARAMS.relevantStatements[value].length
                                ? [DEFAULT_PIPELINE_PARAMS.relevantStatements[value][0]]
                                : [],
                              templates: [...DEFAULT_PIPELINE_PARAMS.domainPatterns[value].baselineTemplates],
                              relevanceOptions: [DEFAULT_PIPELINE_PARAMS.relevanceOptions[0]],
                            }));
                          }
                        }}
                      >
                        <SelectTrigger className="w-full">
                          <SelectValue placeholder="Choose a domain" />
                        </SelectTrigger>
                        <SelectContent>
                          {Object.keys(DEFAULT_PIPELINE_PARAMS.domainPatterns).map(domain => (
                            <SelectItem key={domain} value={domain}>
                              {domain.charAt(0).toUpperCase() + domain.slice(1)}
                            </SelectItem>
                          ))}
                          <SelectItem value="custom">Custom</SelectItem>
                        </SelectContent>
                      </Select>

                    </div>

                    {/* Always show the file upload if domain is custom */}
                    {selectedParams.domain === 'custom' && (
                      <div className="space-y-4">
                        <div className="space-y-1">
                          <Label>Upload Custom Prompts</Label>
                          <Input type="file" onChange={handlePromptUpload} />
                        </div>
                      </div>
                    )}

                    {/* Show standard domain parameters when not custom */}
                    {selectedParams.domain !== 'custom' && (
                      <>
                        {/* Domain-specific Primary Issues */}
                        <div className="space-y-1">
                          <Label>
                            {selectedParams.domain === 'healthcare'
                              ? 'Symptoms'
                              : selectedParams.domain === 'finance'
                                ? 'Financial Issues'
                                : selectedParams.domain === 'education'
                                  ? 'Academic Issues'
                                  : 'Primary Issues'}
                          </Label>
                          <div className="flex flex-wrap gap-1">
                            {DEFAULT_PIPELINE_PARAMS.domainPatterns[selectedParams.domain].primaryIssues.map(
                              issue => (
                                <Badge
                                  key={issue}
                                  variant={
                                    selectedParams.primaryIssues.includes(issue)
                                      ? 'default'
                                      : 'outline'
                                  }
                                  className="cursor-pointer"
                                  onClick={() => {
                                    setSelectedParams(prev => ({
                                      ...prev,
                                      primaryIssues: prev.primaryIssues.includes(issue)
                                        ? prev.primaryIssues.filter(s => s !== issue)
                                        : [...prev.primaryIssues, issue],
                                    }));
                                  }}
                                >
                                  {issue}
                                </Badge>
                              )
                            )}
                          </div>
                        </div>

                        {/* Question Types */}
                        <div className="space-y-1">
                          <Label>Question Types</Label>
                          <div className="flex flex-wrap gap-2">
                            {pipelineParams.questionTypes.map(type => (
                              <Badge
                                key={type}
                                variant={
                                  selectedParams.questionTypes.includes(type)
                                    ? 'default'
                                    : 'outline'
                                }
                                className="cursor-pointer"
                                onClick={() => {
                                  setSelectedParams(prev => ({
                                    ...prev,
                                    questionTypes: prev.questionTypes.includes(type)
                                      ? prev.questionTypes.filter(t => t !== type)
                                      : [...prev.questionTypes, type],
                                  }));
                                }}
                              >
                                {type}
                              </Badge>
                            ))}
                          </div>
                        </div>

                        {/* Recommendation Options */}
                        {selectedParams.questionTypes.some(
                          qt => qt === 'True/False' || qt === 'Multiple Choice'
                        ) && (
                            <div className="space-y-1">
                              <Label>Recommendation Options</Label>
                              <div className="flex flex-wrap gap-1">
                                {DEFAULT_PIPELINE_PARAMS.domainPatterns[selectedParams.domain]
                                  .recommendationPatterns.map(rec => (
                                    <Badge
                                      key={rec}
                                      variant={
                                        selectedParams.recommendations.includes(rec)
                                          ? 'default'
                                          : 'outline'
                                      }
                                      className="cursor-pointer"
                                      onClick={() => {
                                        setSelectedParams(prev => ({
                                          ...prev,
                                          recommendations: prev.recommendations.includes(rec)
                                            ? prev.recommendations.filter(r => r !== rec)
                                            : [...prev.recommendations, rec],
                                        }));
                                      }}
                                    >
                                      {rec}
                                    </Badge>
                                  ))}
                              </div>
                            </div>
                          )}
                      </>
                    )}

                    {/* Always show the Relevant Context options */}
                    <div className="space-y-1">
                      <Label>Relevant Context</Label>
                      <div className="flex flex-wrap gap-2">
                        {pipelineParams.relevanceOptions.map(option => (
                          <Badge
                            key={option}
                            variant={
                              selectedParams.relevanceOptions.includes(option)
                                ? 'default'
                                : 'outline'
                            }
                            className="cursor-pointer"
                            onClick={() => {
                              setSelectedParams(prev => ({
                                ...prev,
                                relevanceOptions: prev.relevanceOptions.includes(option)
                                  ? prev.relevanceOptions.filter(o => o !== option)
                                  : [...prev.relevanceOptions, option],
                              }));
                            }}
                          >
                            {option}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    {/* Show the Relevant Statement Options only if "Relevant" is selected */}
                    {selectedParams.relevanceOptions.includes('Relevant') && (
                      <div className="space-y-1">
                        <Label>Relevant Statement Options</Label>
                        <div className="flex flex-wrap gap-1">
                          {(selectedParams.domain === 'custom'
                            ? customRelevantStatementsRef.current
                            : DEFAULT_PIPELINE_PARAMS.relevantStatements[selectedParams.domain]
                          ).map(statement => (
                            <Badge
                              key={statement}
                              variant={
                                selectedParams.relevantStatements.includes(statement)
                                  ? 'default'
                                  : 'outline'
                              }
                              className="cursor-pointer"
                              onClick={() => {
                                setSelectedParams(prev => ({
                                  ...prev,
                                  relevantStatements: prev.relevantStatements.includes(statement)
                                    ? prev.relevantStatements.filter(s => s !== statement)
                                    : [...prev.relevantStatements, statement],
                                }));
                              }}
                            >
                              {statement}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}


                    {/* Demographics - Always shown */}
                    <div className="space-y-2">
                      <Label>Demographics</Label>
                      <div className="grid grid-cols-2 gap-2">
                        {(Object.keys(pipelineParams.demographics) as Array<
                          keyof typeof pipelineParams.demographics
                        >).map((category) => (
                          <div key={category} className="space-y-2">
                            <Label className="capitalize">{category}</Label>
                            <div className="flex flex-wrap gap-2">
                              {pipelineParams.demographics[category].map((value) => (
                                <Badge
                                  key={value}
                                  variant={
                                    selectedParams.demographics[category].includes(value)
                                      ? 'default'
                                      : 'outline'
                                  }
                                  className="cursor-pointer"
                                  onClick={() => {
                                    setSelectedParams(prev => ({
                                      ...prev,
                                      demographics: {
                                        ...prev.demographics,
                                        [category]: prev.demographics[category].includes(value)
                                          ? prev.demographics[category].filter(v => v !== value)
                                          : [...prev.demographics[category], value],
                                      },
                                    }));
                                  }}
                                >
                                  {value}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Iterations Input */}
                    <div className="space-y-1">
                      <Label className="text-sm font-medium">Iterations per Prompt</Label>
                      <Input
                        type="number"
                        min={1}
                        max={10}
                        value={selectedParams.iterations === 0 ? '' : selectedParams.iterations}
                        onChange={e => {
                          const str = e.target.value
                          const value = str === '' ? '' : parseInt(str, 10)
                          setSelectedParams(prev => ({
                            ...prev,
                            iterations: value as any
                          }))
                        }}
                        onBlur={e => {
                          if (e.target.value === '') {
                            setSelectedParams(prev => ({ ...prev, iterations: 1 }))
                          }
                        }}
                        className="w-full"
                      />
                    </div>
                  </div>
                )}
              </div>


              {/* Action Buttons */}
              <div className="flex flex-wrap gap-2 pt-3">
                <Button
                  onClick={handleAnalyze}
                  disabled={isAnalyzing || isProcessingFile}
                  className="flex-1 h-10 font-medium"
                >
                  {isAnalyzing ? 'Analyzing...' : 'Generate Prompts & Run Analysis'}
                </Button>

                <div className="relative">
                  <Input
                    ref={fileInputRef}
                    type="file"
                    accept=".json"
                    onChange={handleFileUpload}
                    disabled={isAnalyzing || isProcessingFile}
                    className="hidden"
                  />
                  <Input
                    ref={conceptsFileInputRef}
                    type="file"
                    accept=".json"
                    onChange={handleUploadResults}
                    className="hidden"
                  />
                  <Button
                    variant="outline"
                    disabled={isAnalyzing || isProcessingFile || isLoadingConceptsFile}
                    className="flex items-center gap-2 h-10"
                    onClick={handleConceptsUploadClick}
                  >
                    {isProcessingFile || isLoadingConceptsFile ? (
                      'Processing...'
                    ) : (
                      <>
                        <Upload className="h-4 w-4" />
                        Load Results
                      </>
                    )}
                  </Button>
                </div>
              </div>

              {/* Progress Display - Analysis */}
              {progress && (
                <div className="space-y-2 p-4 bg-muted rounded-lg border">
                  <div className="space-y-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">{progress.message}</span>
                      {progress.totalPrompts && (
                        <span className="font-medium">
                          {progress.completedPrompts || 0}/{progress.totalPrompts} prompts
                        </span>
                      )}
                    </div>

                    {progress.totalPrompts && (
                      <div className="w-full bg-secondary rounded-full h-2">
                        <div
                          className="bg-primary rounded-full h-2 transition-all duration-300"
                          style={{
                            width: `${((progress.completedPrompts || 0) / progress.totalPrompts) * 100}%`
                          }}
                        />
                      </div>
                    )}

                    {progress.currentPrompt && (
                      <div className="text-sm text-muted-foreground mt-2">
                        <p className="truncate">Current prompt: {progress.currentPrompt}</p>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Analysis Results section */}
            {analysisResults.length > 0 && (
              <div className="space-y-4 mt-3">
                <div className="border-b pb-1 flex justify-between items-center">
                  <h2 className="text-lg font-semibold tracking-tight">Analysis Results</h2>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={saveAnalysis}
                    className="h-8 px-2 flex items-center gap-1 text-muted-foreground hover:text-foreground"
                  >
                    <Save className="h-3.5 w-3.5" />
                    <Download className="h-3.5 w-3.5" />
                    <span className="text-xs font-medium">Save</span>
                  </Button>
                </div>
                <div className="space-y-2">
                  {analysisResults.map(result => (
                    <Card key={result.id} className="border">
                      <CardContent className="p-4 space-y-3">
                        {/* Result Header */}
                        <div className="flex items-center justify-between pb-2 border-b">
                          <h3 className="font-medium tracking-tight">{result.modelName}</h3>
                        </div>

                        <p className="text-sm text-muted-foreground">
                          {result.details}
                        </p>

                        {/* Demographics Badges */}
                        <div className="flex flex-wrap gap-1 pb-2">
                          {result.demographics.map(demo => (
                            <Badge
                              key={demo}
                              variant="outline"
                              className="rounded-md"
                            >
                              {demo}
                            </Badge>
                          ))}
                        </div>

                        {/* Responses Section */}
                        <div className="space-y-2">
                          <h4 className="font-medium tracking-tight pb-1 border-b">Responses</h4>
                          {result.prompts
                            .slice(
                              (pagination[result.id]?.page || 0) * ITEMS_PER_PAGE,
                              ((pagination[result.id]?.page || 0) + 1) * ITEMS_PER_PAGE
                            )
                            .map((promptResult: PromptResult, idx: number) => {
                              const absoluteIdx = idx + (pagination[result.id]?.page || 0) * ITEMS_PER_PAGE;
                              const isExpanded = pagination[result.id]?.expanded.has(absoluteIdx);

                              return (
                                <div key={idx} className="border rounded-lg p-3 space-y-2">
                                  <div
                                    className="flex items-center justify-between cursor-pointer"
                                    onClick={() => {
                                      setPagination(prev => {
                                        const currentExpanded = new Set(prev[result.id]?.expanded || []);
                                        if (isExpanded) {
                                          currentExpanded.delete(absoluteIdx);
                                        } else {
                                          currentExpanded.add(absoluteIdx);
                                        }
                                        return {
                                          ...prev,
                                          [result.id]: {
                                            page: prev[result.id]?.page || 0,
                                            expanded: currentExpanded
                                          }
                                        };
                                      });
                                    }}
                                  >
                                    <p className="text-sm font-medium">{promptResult.text}</p>
                                    {isExpanded ? (
                                      <ChevronUp className="h-4 w-4 flex-shrink-0" />
                                    ) : (
                                      <ChevronDown className="h-4 w-4 flex-shrink-0" />
                                    )}
                                  </div>

                                  {isExpanded && (
                                    <div className="space-y-1 pt-2">
                                      {promptResult.responses.map((response: string, rIdx: number) => (
                                        <p key={rIdx} className="text-sm text-muted-foreground">
                                          Response {rIdx + 1}: {response}
                                        </p>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              );
                            })}
                        </div>

                        {result.prompts.length > ITEMS_PER_PAGE && (
                          <div className="flex justify-center gap-2 mt-3 pt-2 border-t">
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => {
                                setPagination(prev => ({
                                  ...prev,
                                  [result.id]: {
                                    page: Math.max(0, (prev[result.id]?.page || 0) - 1),
                                    expanded: prev[result.id]?.expanded || new Set()
                                  }
                                }));
                              }}
                              disabled={(pagination[result.id]?.page || 0) === 0}
                            >
                              Previous
                            </Button>

                            <span className="flex items-center text-sm text-muted-foreground">
                              Page {(pagination[result.id]?.page || 0) + 1} of{' '}
                              {Math.ceil(result.prompts.length / ITEMS_PER_PAGE)}
                            </span>

                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => {
                                setPagination(prev => ({
                                  ...prev,
                                  [result.id]: {
                                    page: Math.min(
                                      Math.ceil(result.prompts.length / ITEMS_PER_PAGE) - 1,
                                      (prev[result.id]?.page || 0) + 1
                                    ),
                                    expanded: prev[result.id]?.expanded || new Set()
                                  }
                                }));
                              }}
                              disabled={
                                (pagination[result.id]?.page || 0) >=
                                Math.ceil(result.prompts.length / ITEMS_PER_PAGE) - 1
                              }
                            >
                              Next
                            </Button>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Right Column - Concept Extraction Results */}
          {(conceptData.concepts.size > 0 || ldaResults || embeddingsResults.length > 0) && (
            <div className="flex-1 border-t sm:border-t-0 sm:border-l pt-4 sm:pt-0 sm:pl-6 flex flex-col h-[auto] sm:h-[100dvh] overflow-hidden">
              <div className="flex flex-col h-full">
                <div className="p-4 sm:p-6 pb-4">
                  <div className="border-b pb-1 mb-4">
                    <div className="flex justify-between items-center">
                      <h2 className="text-lg font-semibold tracking-tight">Concept Analysis</h2>
                      <div className="flex gap-2">
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => {
                                  if (!conceptData.extractedConcepts || !ldaResults || !embeddingsResults.length) {
                                    toast.error('Missing required data for merged analysis');
                                    return;
                                  }
                                  const csv = createMergedAnalysisCSV(
                                    analysisResults,
                                    conceptData.extractedConcepts,
                                    ldaResults,
                                    embeddingsResults,
                                    conceptData.clusters?.all || []
                                  );
                                  downloadCSV(csv, 'merged_analysis.csv');
                                }}
                                disabled={!conceptData.extractedConcepts || !ldaResults || !embeddingsResults.length}
                              >
                                <BarChart3 className="h-4 w-4 mr-2" />
                                Prompt Analysis CSV
                              </Button>
                            </TooltipTrigger>
                            <TooltipContent>Download Prompt Analysis CSV</TooltipContent>
                          </Tooltip>
                        </TooltipProvider>

                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={handleDownloadResults}
                                disabled={!conceptData.concepts.size && !ldaResults && !embeddingsResults.length}
                              >
                                <Download className="h-4 w-4" /> JSON
                              </Button>
                            </TooltipTrigger>
                            <TooltipContent>Download Results</TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      </div>
                    </div>
                  </div>
                  <ExtractionProgressDisplay />
                </div>
                {/* Tabs Container */}
                <div className="flex-1 min-h-0">
                  <Tabs defaultValue="llm" className="w-full h-full flex flex-col">
                    <div className="px-4 sm:px-6">
                      <TabsList className="grid w-full lg:grid-cols-4 md:grid-cols-2 sm:grid-cols-1 mb-4 h-auto">
                        <TabsTrigger value="llm" className="py-2">LLM Concepts</TabsTrigger>
                        <TabsTrigger value="lda" className="py-2">LDA Concepts</TabsTrigger>
                        <TabsTrigger value="embeddings" className="py-2">Embeddings Concepts</TabsTrigger>
                        <TabsTrigger value="agreement" className="py-2">Agreement Scores</TabsTrigger>
                      </TabsList>
                    </div>

                    {/* LLM Concepts Tab */}
                    <TabsContent value="llm" className="flex-1 overflow-y-auto px-4 sm:px-6 min-h-0">
                      {conceptData.concepts.size > 0 && (
                        <>
                          {conceptData.clusters ? (
                            <div className="space-y-6 pb-6">
                              {/* Mobile view */}
                              <div className="sm:hidden">
                                <div className="space-y-4">
                                  {Array.from(conceptData.concepts.entries()).map(([concept, count]) => (
                                    <Card key={concept} className="p-4">
                                      <div className="flex justify-between items-center">
                                        <span className="font-medium">{concept}</span>
                                        <Badge>{count}</Badge>
                                      </div>
                                    </Card>
                                  ))}
                                </div>
                              </div>
                              {/* Desktop view */}
                              <div className="hidden sm:block">
                                <ConceptVisualizations
                                  conceptData={{
                                    concepts: conceptData.concepts,
                                    demographicDistributions: Object.fromEntries(
                                      Array.from(conceptData.demographicDistributions.entries()).map(([key, value]) => [
                                        key,
                                        Object.fromEntries(
                                          Array.from(value.entries()).map(([subgroup, dist]) => [
                                            subgroup,
                                            Array.from(dist.values()).map(count => [count]),
                                          ])
                                        ),
                                      ])
                                    ),
                                    clusters: {
                                      all: conceptData.clusters?.all?.map((cluster) => ({
                                        id: cluster.id,
                                        concepts: cluster.concepts,
                                        frequency: cluster.frequency,
                                        label: cluster.label,
                                        total_frequency:
                                          cluster.total_frequency !== undefined
                                            ? cluster.total_frequency
                                            : Array.isArray(cluster.frequency)
                                              ? cluster.frequency.reduce((a, b) => a + b, 0)
                                              : 0,
                                      })) ?? [],
                                      demographics: conceptData.clusters?.demographics
                                        ? Object.fromEntries(
                                          Object.entries(conceptData.clusters.demographics).map(([demo, clusters]) => [
                                            demo,
                                            Array.isArray(clusters)
                                              ? clusters.map((cluster) => ({
                                                id: cluster.id,
                                                concepts: cluster.concepts,
                                                frequency: cluster.frequency,
                                                label: cluster.label ? cluster.label : cluster.id.toString(),
                                                total_frequency:
                                                  cluster.total_frequency !== undefined
                                                    ? cluster.total_frequency
                                                    : Array.isArray(cluster.frequency)
                                                      ? cluster.frequency.reduce((a, b) => a + b, 0)
                                                      : 0,
                                              }))
                                              : []
                                          ])
                                        )
                                        : {},
                                    },
                                    rawResults: conceptData.rawResults,
                                    extractedConcepts: conceptData.extractedConcepts,
                                  }}
                                />
                              </div>
                            </div>
                          ) : (
                            <div className="flex justify-center items-center h-full">
                              <p>Processing data... Please wait.</p>
                            </div>
                          )}
                        </>
                      )}
                    </TabsContent>
                    <TabsContent value="lda" className="flex-1 overflow-y-auto px-4 sm:px-6 min-h-0">
                      {ldaResults && analysisResults ? (
                        <div className="space-y-6 pb-6">
                          {/* Existing Topics Display */}
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                            {ldaResults.topics.map((topic) => (
                              <Card key={topic.topic_id} className="p-4">
                                <h4 className="font-medium mb-2">Topic {topic.topic_id}</h4>
                                <div className="space-y-2">
                                  <div className="flex flex-wrap gap-2">
                                    {topic.words.map((word, idx) => (
                                      <Badge key={`${word}-${idx}`} variant="secondary" className="flex items-center gap-1">
                                        <span>{word}</span>
                                        <span className="text-xs opacity-70">
                                          {topic.weights && !isNaN(topic.weights[idx])
                                            ? (topic.weights[idx] * 100).toFixed(1) + '%'
                                            : 'N/A'}
                                        </span>
                                      </Badge>
                                    ))}
                                  </div>
                                </div>
                              </Card>
                            ))}
                          </div>
                          {/* Visualizations */}
                          <LDAVisualizations ldaResults={ldaResults} />
                        </div>
                      ) : (
                        <div className="flex justify-center items-center h-full">
                          <p>Processing data... Please wait.</p>
                        </div>
                      )}
                    </TabsContent>



                    {/* Embeddings Tab - Mobile optimized */}
                    <TabsContent value="embeddings" className="flex-1 overflow-y-auto px-4 sm:px-6 min-h-0">
                      {embeddingsResults.length > 0 && (
                        <div className="w-full pb-6">
                          <div className="hidden sm:block">
                            {/* Desktop-only view */}
                            <EmbeddingsVisualizations
                              results={embeddingsResults}
                              analysisResults={analysisResults}
                            />
                          </div>
                        </div>
                      )}
                    </TabsContent>

                    {/* Agreement Scores Tab - Mobile optimized */}
                    <TabsContent value="agreement" className="flex-1 overflow-y-auto px-4 sm:px-6 min-h-0">
                      {conceptData.concepts.size > 0 && ldaResults && embeddingsResults.length > 0 && (
                        <div className="space-y-6 pb-6">
                          <AgreementScoreVisualizations
                            agreementData={agreementData}
                            embeddingsData={embeddingsResults.map((result) => ({
                              pca_one: result.coordinates?.[0]?.[0] ?? 0,
                              pca_two: result.coordinates?.[0]?.[1] ?? 0
                            }))}
                          />
                        </div>
                      )}
                    </TabsContent>
                  </Tabs>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
