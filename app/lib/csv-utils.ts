import { AnalysisResult, ExtractedConcepts, LDATopicResult } from "@/app/types/pipeline";
import Papa from 'papaparse';
import * as lemmatizer from 'wink-lemmatizer';

export type ConceptExtractionRow = {
  Category: string;
  Relevance: string; 
  Perspective: string;
  Question_Type: string;
  Prompt: string;
  Gender: string | null;
  Age: string;
  Race: string;
  Socioeconomic: string;
  Response: string;
  GPT_Categories: string;
  Cluster: string;
}

export type LDAExtractionRow = {
  Prompt: string;
  Response: string;
  Gender: string | null;
  Age: string;
  Race: string;
  Socioeconomic: string;
  Dominant_Topic: number;
  Topic_Probability: number;
  Topic_Keywords: string;
  Topic_Description: string;
  Topic_Distribution?: string;
}

export type EmbeddingsExtractionRow = {
  Prompt: string;
  Response: string;
  processed_response: string;
  pca_one: number;
  pca_two: number;
  cluster: number;
  pca_cluster_number: string;
  raw_embeddings: number[];
  Gender: string | null;
  Age: string;
  Race: string;
  Socioeconomic: string;
}

type ClusterData = {
  id: number;
  concepts: string[];
  frequency: number[];
}

type MergedRow = {
  Category: string;
  Relevance: string;
  Perspective: string;
  Question_Type: string;
  Prompt: string;
  Gender: string | null;
  Age: string;
  Race: string;
  Socioeconomic: string;
  Response: string;
  GPT_Categories: string;
  Concept_Cluster: string;
  Dominant_Topic: number | string;
  Topic_Probability: number | string;
  Topic_Keywords: string;
  Topic_Distribution: string;
  PCA_One: number | string;
  PCA_Two: number | string;
  Embeddings_Cluster: number | string;
  Raw_Embeddings: number[] | string;
}

function extractDemographics(demographics: string[]): {
  gender: string | null;
  age: string;
  race: string;
  socioeconomic: string;
} {
  const gender = demographics.find(d => ['woman', 'man', 'non-binary'].includes(d)) || "Unknown";
  const age = demographics.find(d => ['Young Adult', 'Middle-aged', 'Elderly'].includes(d)) || "Unknown";
  const race = demographics.find(d => ['Asian', 'Black', 'Hispanic', 'White', 'Other'].includes(d)) || "Unknown";
  const socioeconomic = demographics.find(d => ['Low income', 'Middle income', 'High income'].includes(d)) || "Unknown";
  return { gender, age, race, socioeconomic };
}

function normalizeConcept(concept: string): string {
  return concept
    .toLowerCase()
    .replace(/[^\w\s]/g, '')                  
    .split(/\s+/)                             
    .map(token => lemmatizer.noun(token))
    .join(' ')
    .trim();
}

export function createConceptExtractionCSV(
  analysisResults: AnalysisResult[],
  extractedConcepts: ExtractedConcepts[], 
  clusters: ClusterData[]
): string {
  const rows: ConceptExtractionRow[] = [];

  analysisResults.forEach(result => {
    result.prompts.forEach(prompt => {
      const { gender, age, race, socioeconomic } = extractDemographics(prompt.metadata.demographics);
      prompt.responses.forEach(response => {
        const matchingConcepts = extractedConcepts.find(
          ec => ec.response === response.replace(/[\n\r]+/g, ' ').trim()
        );

        if (matchingConcepts) {
          const concepts = Array.isArray(matchingConcepts.concepts) 
            ? matchingConcepts.concepts 
            : JSON.parse(matchingConcepts.concepts as unknown as string);

          concepts.forEach((concept: string) => {
            const normConcept = normalizeConcept(concept);
            // Try to find a matching cluster using our normalized value.
            const foundCluster = clusters?.find(c =>
              Array.isArray(c.concepts) &&
              c.concepts.some(cConcept => normalizeConcept(cConcept) === normConcept)
            );
            // If no match is found, assign a fallback cluster (e.g. "unclustered").
            const clusterNumber = foundCluster ? foundCluster.id.toString() : "unclustered";

            rows.push({
              Category: "Anxiety Management",
              Relevance: "Neutral",
              Perspective: prompt.metadata.perspective || "First",
              Question_Type: "Open-Ended",
              Prompt: prompt.text,
              Gender: gender,
              Age: age,
              Race: matchingConcepts.demographics?.find(d => d.category === 'ethnicities')?.value || race,
              Socioeconomic: socioeconomic,
              Response: response,
              GPT_Categories: concept,
              Cluster: clusterNumber
            });
          });
        }
      });
    });
  });

  return Papa.unparse(rows, {
    quotes: true,
    delimiter: ",",
    header: true
  });

}
export function createLDAExtractionCSV(
  analysisResults: AnalysisResult[],
  ldaResults: {
    topics: LDATopicResult[];
    distributions: number[][];
  }
): string {
  const rows: LDAExtractionRow[] = [];
  let responseIndex = 0;

  analysisResults.forEach((result) => {
    result.prompts.forEach((prompt) => {
      const { gender, age, race, socioeconomic } = extractDemographics(prompt.metadata.demographics);
      prompt.responses.forEach((response) => {
      
        const topicDistribution = ldaResults.distributions[responseIndex];
        
        // If no distribution, skip the response
        if (!topicDistribution) {
          console.warn(`No topic distribution found for response at index ${responseIndex}`);
          responseIndex++;
          return;
        }

        const dominantTopicIndex = topicDistribution.indexOf(Math.max(...topicDistribution));
        const dominantTopic = ldaResults.topics[dominantTopicIndex];
        const topicProbability = topicDistribution[dominantTopicIndex];
        const topKeywords = dominantTopic.words.slice(0, 5).join(', ');
        const topicDescription = ldaResults.topics
          .map(
            (topic, idx) =>
              `Topic ${topic.topic_id} (${topicDistribution[idx].toFixed(3)})`
          )
          .join('; ');

        rows.push({
          Prompt: prompt.text,
          Response: response,
          Gender: gender,
          Age: age,
          Race: race,
          Socioeconomic: socioeconomic,
          Dominant_Topic: dominantTopic.topic_id,
          Topic_Probability: topicProbability,
          Topic_Keywords: topKeywords,
          Topic_Description: topicDescription,
          Topic_Distribution: JSON.stringify(
            ldaResults.topics.map((topic, idx) => ({
              topic_id: topic.topic_id,
              probability: topicDistribution[idx],
            }))
          ),
        });

        responseIndex++; 
      });
    });
  });

  return Papa.unparse(rows, {
    quotes: true,
    delimiter: ",",
    header: true,
  });
}



export function createEmbeddingsExtractionCSV(
  analysisResults: AnalysisResult[],
  embeddingsResults: {
    cluster_id: number;
    representative_responses: string[];
    coordinates: number[][];
    embeddings: number[][];
  }[]
): string {
  const rows: EmbeddingsExtractionRow[] = [];

  analysisResults.forEach(result => {
    result.prompts.forEach(prompt => {
      const { gender, age, race, socioeconomic } = extractDemographics(prompt.metadata.demographics);
      prompt.responses.forEach(response => {
        const cluster = embeddingsResults.find(c => 
          c.representative_responses.some(rep => 
            rep.replace(/[\n\r\s]+/g, ' ').trim().toLowerCase() === 
            response.replace(/[\n\r\s]+/g, ' ').trim().toLowerCase()
          )
        );
        
        if (cluster) {
          const responseIdx = cluster.representative_responses.findIndex(rep =>
            rep.replace(/[\n\r\s]+/g, ' ').trim().toLowerCase() === 
            response.replace(/[\n\r\s]+/g, ' ').trim().toLowerCase()
          );

          if (responseIdx !== -1) {
            const coordinates = cluster.coordinates[responseIdx];
            const embeddings = cluster.embeddings[responseIdx];

            rows.push({
              Prompt: prompt.text,
              Response: response,
              processed_response: response.toLowerCase().replace(/[^\w\s]/g, ''),
              pca_one: coordinates[0],
              pca_two: coordinates[1],
              cluster: Math.round(cluster.cluster_id),
              pca_cluster_number: `Cluster ${cluster.cluster_id + 1}`,
              raw_embeddings: embeddings,
              Gender: gender,
              Age: age,
              Race: race,
              Socioeconomic: socioeconomic
            });
          }
        }
      });
    });
  });

  return Papa.unparse(rows, {
    quotes: true,
    delimiter: ",",
    header: true
  });
}

export function createMergedAnalysisCSV(
  analysisResults: AnalysisResult[],
  extractedConcepts: ExtractedConcepts[],
  ldaResults: {
    topics: LDATopicResult[];
    distributions: number[][];
  },
  embeddingsResults: {
    cluster_id: number;
    representative_responses: string[];
    coordinates: number[][];
    embeddings: number[][];
    size: number;
    distribution: { [key: string]: number };
  }[],
  clusters: ClusterData[]
): string {
  const mergedRows: MergedRow[] = [];
  let currentResponseIdx = 0;

  for (const result of analysisResults) {
    for (const prompt of result.prompts) {
      for (const response of prompt.responses) {
        // If we have run out of distributions, skip this response.
        if (currentResponseIdx >= ldaResults.distributions.length) {
          console.warn(`No topic distribution for response at index ${currentResponseIdx}`);
          continue;
        }
        const topicDistribution = ldaResults.distributions[currentResponseIdx];
        if (!topicDistribution || topicDistribution.length !== ldaResults.topics.length) {
          console.warn(`Invalid topic distribution for response at index ${currentResponseIdx}`);
          currentResponseIdx++;
          continue;
        }

        const dominantTopicIndex = topicDistribution.indexOf(Math.max(...topicDistribution));
        const dominantTopic = ldaResults.topics[dominantTopicIndex];
        const topicProbability = topicDistribution[dominantTopicIndex];
        const topKeywords = dominantTopic.words.slice(0, 5).join(', ');

        // Find a matching embedding cluster, if any.
        const embeddingCluster = embeddingsResults.find(c =>
          c.representative_responses.some(rep =>
            rep.replace(/[\n\r\s]+/g, ' ').trim().toLowerCase() === response.replace(/[\n\r\s]+/g, ' ').trim().toLowerCase()
          )
        );
        const responseEmbeddingIdx = embeddingCluster?.representative_responses.findIndex(rep =>
          rep.replace(/[\n\r\s]+/g, ' ').trim().toLowerCase() === response.replace(/[\n\r\s]+/g, ' ').trim().toLowerCase()
        );

        const matchingConcepts = extractedConcepts.find(
          ec => ec.response === response.replace(/[\n\r]+/g, ' ').trim()
        );

        const { gender, age, race, socioeconomic } = extractDemographics(prompt.metadata.demographics);

        if (matchingConcepts) {
          const concepts = Array.isArray(matchingConcepts.concepts)
            ? matchingConcepts.concepts
            : JSON.parse(matchingConcepts.concepts as unknown as string);

          for (const concept of concepts) {
            const normConcept = normalizeConcept(concept);
            const conceptClusterMap = clusters.reduce((acc, cluster) => {
              cluster.concepts.forEach(cConcept => {
                acc[normalizeConcept(cConcept)] = cluster.id.toString();
              });
              return acc;
            }, {} as Record<string, string>);

            const clusterNumber = conceptClusterMap[normConcept] || "unclustered";

            mergedRows.push({
              Category: "Anxiety Management",
              Relevance: "Neutral",
              Perspective: prompt.metadata.perspective || "First",
              Question_Type: "Open-Ended",
              Prompt: prompt.text,
              Gender: gender,
              Age: age,
              Race: matchingConcepts.demographics?.find(d => d.category === 'ethnicities')?.value || race,
              Socioeconomic: socioeconomic,
              Response: response,
              GPT_Categories: concept,
              Concept_Cluster: clusterNumber,
              Dominant_Topic: dominantTopic?.topic_id ?? "",
              Topic_Probability: dominantTopic ? topicProbability : "",
              Topic_Keywords: dominantTopic ? topKeywords : "",
              Topic_Distribution: JSON.stringify(
                ldaResults.topics.map((topic, idx) => ({
                  topic_id: topic.topic_id,
                  probability: topicDistribution[idx],
                }))
              ),
              PCA_One: embeddingCluster && responseEmbeddingIdx !== undefined && responseEmbeddingIdx !== -1
                ? embeddingCluster.coordinates[responseEmbeddingIdx][0]
                : "",
              PCA_Two: embeddingCluster && responseEmbeddingIdx !== undefined && responseEmbeddingIdx !== -1
                ? embeddingCluster.coordinates[responseEmbeddingIdx][1]
                : "",
              Embeddings_Cluster: embeddingCluster ? embeddingCluster.cluster_id : "",
              Raw_Embeddings: embeddingCluster && responseEmbeddingIdx !== undefined && responseEmbeddingIdx !== -1
                ? embeddingCluster.embeddings[responseEmbeddingIdx]
                : []
            });
          }
        }
        currentResponseIdx++;
      }
    }
  }

  return Papa.unparse(mergedRows, {
    quotes: true,
    delimiter: ",",
    header: true
  });
}

export function downloadCSV(csv: string, filename: string) {
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
