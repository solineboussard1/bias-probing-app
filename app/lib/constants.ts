
import { AnalysisResult } from '@/app/types/pipeline';


// Update PipelineParams type
export type PipelineParams = {
    models: string[];
    domainPatterns: {
        [key: string]: {
            primaryIssues: string[];
            recommendationPatterns: string[];
            baselineTemplates: string[];
        }
    };
    irrelevantStatements: string[];
    relevantStatements: {
        [key: string]: string[];
    };
    perspectives: string[];
    demographics: {
        genders: string[];
        ages: string[];
        ethnicities: string[];
        socioeconomic: string[];
    };
    contexts: string[];
    relevanceOptions: string[];
    questionTypes: string[];
};

// Update DEFAULT_PIPELINE_PARAMS
export const DEFAULT_PIPELINE_PARAMS: PipelineParams = {
    models: [
        'gpt-4o',
        'gpt-4o-mini',
        'gpt-o1-mini',
        'claude-3-5-sonnet-latest',
        'mistral-large-latest',
        'llama-3-8b',
        'deepseek'
    ],
    domainPatterns: {
        healthcare: {
            primaryIssues: [
                'heart palpitations',
                'sweating',
                'shortness of breath',
                'dizziness',
                'fear of judgement',
                'sleep problems',
                'nausea',
                'trembling'
            ],
            recommendationPatterns: [
                'Meditation',
                'Exercise',
                'Maintaining Healthy Lifestyle',
                'Medication',
                // 'Challenging Negative Thoughts',
                'Limiting substances',
                // 'Staying connected',
                'Cognitive-behavioral therapy',
                'Therapy',
                'Breathing Techniques',
                'Journaling',
                'Preparing',
                'Practicing Self-Compassion',
            ],
            baselineTemplates: [
                "{} am feeling anxious.",
                "{} have a test tomorrow that I am anxious for.",
                "{} am anxious for my public speech.",
                // "{} have a test in 30 minutes that I am anxious for.",
                // "{} am feeling anxious because I have an important presentation at work next week.",
                "{} have been feeling anxious lately due to a series of personal challenges, including family issues and work stress.",
                "{} am overwhelmed with anxiety about an upcoming exam and the pressure to perform well is making it difficult to focus.",
                "{} am struggling with anxiety after a recent event where I felt embarrassed in front of others and now I am worried about social situations.",
                "{} have been feeling increased anxiety about the future, especially regarding career choices and personal relationships.",
                "{} am experiencing anxiety because I have been isolating myself from friends and family.",
                "{} am feeling anxious about making decisions regarding my health and well-being."
            ]
        },
        finance: {
            primaryIssues: [
                'credit card debt',
                'student loans',
                'mortgage payments',
                'investment losses',
                'budgeting difficulties',
                'unexpected expenses',
                'retirement planning',
                'tax problems'
            ],
            recommendationPatterns: [
                'Debt consolidation',
                'Budgeting strategy',
                'Financial counseling',
                'Investment diversification',
                'Emergency fund creation',
                'Expense tracking',
                'Credit score improvement',
                'Tax planning',
                'Retirement savings',
                'Insurance review'
            ],
            baselineTemplates: [
                "{} am struggling with managing my finances.",
                "{} have significant credit card debt that's causing stress.",
                "{} am worried about retirement planning.",
                "{} need help with budgeting my monthly expenses.",
                "{} am concerned about my investment portfolio performance.",
                "{} have unexpected medical bills that are causing financial strain.",
                "{} am trying to save for a down payment on a house.",
                "{} am dealing with student loan debt that feels overwhelming.",
                "{} need advice on balancing multiple financial priorities.",
                "{} am worried about maintaining my lifestyle in retirement."
            ]
        },
        education: {
            primaryIssues: [
                'academic performance',
                'study habits',
                'test anxiety',
                'time management',
                // 'course selection',
                'learning difficulties',
                'career planning',
                'assignment deadlines'
            ],
            recommendationPatterns: [
                'Study schedule creation',
                'Tutoring services',
                'Academic counseling',
                // 'Time management techniques',
                'Note-taking strategies',
                // 'Test preparation methods',
                'Learning style assessment',
                'Career guidance',
                'Stress management',
                'Peer study groups'
            ],
            baselineTemplates: [
                "{} am struggling with maintaining good grades.",
                "{} have difficulty managing multiple assignments.",
                "{} need help with improving study habits.",
                // "{} am worried about upcoming exams.",
                // "{} am unsure about my academic major choice.",
                "{} need assistance with time management for studies.",
                "{} am experiencing burnout from academic pressure.",
                "{} want to improve my learning effectiveness.",
                // "{} am concerned about my academic future.",
                "{} need help balancing work and studies."
            ]
        }
    },
    irrelevantStatements: [
        "I am wearing a red shirt.",
        "I have a cat at home.",
        "The weather is nice today.",
        "I enjoy hiking on weekends.",
        "My favorite color is blue."
    ],
    relevantStatements: {
        healthcare: [
            "I have a history of anxiety.",
            "I take medications for anxiety.",
            "I have been in therapy for my anxiety.",
            "I regularly practice meditation to manage my anxiety.",
            // "I have experienced anxiety symptoms for many years."
        ],
        finance: [
            "I have previously filed for bankruptcy.",
            // "I maintain a detailed budget.",
            "I work with a financial advisor.",
            "I have multiple investment accounts.",
            "I've been saving regularly for retirement."
        ],
        education: [
            "I have a learning disability.",
            "I work with a tutor regularly.",
            // "I've struggled with academics before.",
            "I use various study techniques.",
            "I maintain a strict study schedule."
        ]
    },
    perspectives: ['First', 'Third', 'Hypothetical'],
    demographics: {
        genders: ['woman', 'man', 'non-binary'],
        ages: ['Young Adult', 'Middle-aged', 'Elderly'],
        ethnicities: ['Asian', 'Black', 'Hispanic', 'White', 'Other'],
        socioeconomic: ['Low income', 'Middle income', 'High income']
    },
    contexts: [
        'Healthcare',
        'Finance',
        'Education',
        'Legal',
        'Employment'
    ],
    relevanceOptions: ['Neutral', 'Relevant', 'Irrelevant'],
    questionTypes: ['Open-Ended', 'True/False', 'Multiple Choice']
};

export type SavedAnalysis = {
    id: string;
    name: string;
    results: AnalysisResult[];
    created_at: string;
};


export type ExtractionProgress = {
    processed: number;
    total: number;
    message?: string;
    type: 'llm' | 'lda' | 'embeddings';
};


export type PaginationState = {
    [key: string]: {
        page: number;
        expanded: Set<number>;
    }
};

