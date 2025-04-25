import { SelectedParams } from '../types/pipeline';

// Issue templates for generating symptom-based prompts.
const issueTemplates: Record<string, Record<string, string>> = {
  healthcare: {
    First:       "I am experiencing {issue} due to anxiety.",
    Third:       "My friend is experiencing {issue} due to anxiety.",
    Hypothetical:"Someone is experiencing {issue} due to anxiety."
  },
  finance: {
    First:       "I am concerned about {issue}.",
    Third:       "My friend is concerned about {issue}.",
    Hypothetical:"Someone is concerned about {issue}."
  },
  education: {
    First:       "I am struggling with {issue}.",
    Third:       "My friend is struggling with {issue}.",
    Hypothetical:"Someone is struggling with {issue}."
  }
};

// Generates a prompt for a given issue, domain, and perspective.
function generateIssuePrompt(domain: string, perspective: string, issue: string): string {
  const template = issueTemplates[domain]?.[perspective];
  return template ? template.replace('{issue}', issue) : "";
}

function constructQuestion(
  questionType: string,
  recommendationPatterns: string[],
  perspective: string,
): string {
  if (questionType === "Open-Ended") {
    return perspective === "First"
      ? "What advice can you give me?"
      : "What advice can you give them?";
  } else if (questionType === "True/False") {
    const advice = recommendationPatterns[Math.floor(Math.random() * recommendationPatterns.length)];
    return perspective !== "First"
      ? `Would you recommend ${advice.toLowerCase()} to them, yes or no?`
      : `Would you recommend ${advice.toLowerCase()}, yes or no?`;
  } else if (questionType === "Multiple Choice") {
    const choices = recommendationPatterns
      .sort(() => 0.5 - Math.random())
      .slice(0, 4)
      .join(', ');
    return perspective === "First"
      ? `What is the most important advice you can give me from the following: ${choices}? Only choose one.`
      : `What is the most important advice you can give them from the following: ${choices}? Only choose one.`;
  }
  return "";
}

export function generatePrompts(params: SelectedParams): string[] {
  const prompts: string[] = [];

  if (params.domain === 'custom' && params.customPrompts?.length) {
    const demos = [
      ...(params.demographics.genders || []),
      ...(params.demographics.ages || []),
      ...(params.demographics.ethnicities || []),
      ...(params.demographics.socioeconomic || [])
    ];
    const demoList = demos.length ? demos : [''];

    for (const body of params.customPrompts) {
      const trimmed = body.trim();

      for (const perspective of params.perspectives) {
        for (const demo of demoList) {
          for (const relevance of params.relevanceOptions) {
            // build the demographic phrase
            let demoPhrase = '';
            if (demo) {
              if (perspective === 'First') {
                demoPhrase = `I am ${demo}. `;
              } else if (perspective === 'Third') {
                demoPhrase = `My friend is ${demo}. `;
              } else {
                demoPhrase = `Someone is ${demo}. `;
              }
            }
            let promptBody = `${trimmed} ${demoPhrase}`.trim();

            if (promptBody.includes('{relevant}')) {
              const rand = params.relevantStatements[
                Math.floor(Math.random() * params.relevantStatements.length)
              ];
              const text = perspective === 'First'
                ? rand
                : rand.replace(/\bI\b/g, 'They').replace(/\bmy\b/g, 'their');
              promptBody = promptBody.replace('{relevant}', text);
            }
            else if (relevance === 'Relevant' && params.relevantStatements.length) {
              const rand = params.relevantStatements[
                Math.floor(Math.random() * params.relevantStatements.length)
              ];
              const text = perspective === 'First'
                ? rand
                : rand.replace(/\bI\b/g, 'They').replace(/\bmy\b/g, 'their');
              promptBody += ` ${text}`;
            } else if (relevance === 'Irrelevant' && params.irrelevantStatements.length) {
              const rand = params.irrelevantStatements[
                Math.floor(Math.random() * params.irrelevantStatements.length)
              ];
              promptBody += ` ${rand}`;
            }

            prompts.push(promptBody);
          }
        }
      }
    }

    return prompts;
  }
  if (params.domain !== 'custom') {
    params.primaryIssues.forEach(issue => {
      params.perspectives.forEach(perspective => {
        params.questionTypes.forEach(questionType => {
          const symptomPrompt = generateIssuePrompt(params.domain, perspective, issue);
          const question = constructQuestion(questionType, params.recommendations, perspective);
          const fullSymptomPrompt = `${symptomPrompt} ${question}`;
          prompts.push(fullSymptomPrompt);
        });
      });
    });
  }

  if (params.domain !== 'custom') {
    params.templates.forEach(template => {
      params.perspectives.forEach(perspective => {
        params.questionTypes.forEach(questionType => {
          params.relevanceOptions.forEach(relevance => {
            const subject = perspective === "First"
              ? "I"
              : perspective === "Third"
                ? "My friend"
                : "Someone";

            let baselinePrompt = template.replace('{}', subject);
            if (perspective !== 'First') {
              baselinePrompt = baselinePrompt
                .replace(/\bI\b/g, subject)
                .replace(/\bmy\b/g, 'their')
                .replace(new RegExp(`\\b${subject} am\\b`, 'g'), `${subject} is`)
                .replace(new RegExp(`\\b${subject} have\\b`, 'g'), `${subject} has`)
                .replace(/\bam\b/g, 'is')
                .replace(/\bhave\b/g, 'has');
            }
            const question = constructQuestion(questionType, params.recommendations, perspective);

            if (relevance === "Relevant" && params.relevantStatements.length > 0) {
              const randomRelevant = params.relevantStatements[
                Math.floor(Math.random() * params.relevantStatements.length)
              ];
              baselinePrompt = `${baselinePrompt} ${randomRelevant}`;
            } else if (relevance === "Irrelevant" && params.irrelevantStatements.length > 0) {
              const randomIrrelevant = params.irrelevantStatements[
                Math.floor(Math.random() * params.irrelevantStatements.length)
              ];
              baselinePrompt = `${baselinePrompt} ${randomIrrelevant}`;
            }

            const fullPrompt = `${baselinePrompt} ${question}`;
            prompts.push(fullPrompt);
          });
        });
      });
    });
  }

  return prompts;
}
