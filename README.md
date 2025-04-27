
# Bias Probing Tool

# Overview 
This tool is a dynamic analysis tool designed to help users understand the sensitvity and output spaces of LLMs in their domains.

### Key Capabilities

- **Prompt Generation:**
  Dynamically builds queries based on a combination of user-selected parameters, such as perspective, question type, and demographic details. For non-custom domains, pre-designed templates drive the prompt creation. When the custom domain is chosen, users supplies prompts and relevant information.
- **Model Integration:**
  Supports multiple providers (OpenAI, Anthropic, HuggingFace, Deepseek, Mistral) by mapping each chosen model through the users API keys. 
- **Concept Extraction and Batch Processing:**  
  Leverages pipeline collects responses and groups these responses based on demographic segments. Then multiple machine learning techniques are run to analyze the text including: 
  - **LLM-based extraction:** Extracts relevant concepts from responses.
  - **Topic clustering (LDA):** Generates topic clusters to highlight recurring themes.
  - **Embeddings extraction:** Analyzes semantic response relationships for deeper insights.

## How to Use the Tool
### 1. Configure Your Analysis

1. **Select a Model & Domain**  
   - Choose from supported models (e.g., GPT variants, Claude, Mistral).  
   - Pick a domain: healthcare_, finance_, education_ or custom.

2. **Define Parameters**  
   - ​For pre‑defined domains: select primary issues, question types (Open‑Ended, True/False, Multiple Choice), relevance options (Relevant/Irrelevant), recommendation patterns (if true/false or multiple choice is selected), relevant statements (if relevant category is selected) demographic filters (gender, age, ethnicity, socioeconomic).
   - ​For **custom** domain: you’ll upload your own prompts (see next step).

3. **Upload Your Own Prompts (Custom Domain only)**  
   - Click the **Upload Prompts** button in the sidebar.  
   - Provide a JSON file in one of these formats:
     ```json
     [
       "My custom prompt about stress management.",
       "Another custom prompt on workplace anxiety."
     ]
     ```
     or
     ```json
     {
       "prompts": [
         "First custom prompt text.",
         "Second custom prompt text."
       ],
       "relevantStatements": [
         "I feel overwhelmed at work.",
         "I take medication for anxiety."
       ]
     }
     ```
   - The tool will parse the file, populate `customPrompts`, and (if provided) import `relevantStatements` for use in relevance statements.

### 2. Start the Analysis

- **Initiate the Pipeline:**  
  Once your settings are configured, trigger the analysis. The tool immediately begins generating prompt templates and combining them with selected demographic groups.

- **Streamed Updates:**  
  As the analysis runs, you will receive real-time progress updates. These updates inform you about various stages such as prompt generation, execution progress for each prompt, and ongoing extractions.

- **Download and Save:**  
  You have the option to save your analysis as a JSON file. This allows you to review the data and reload the analysis later for further investigation.


### 3. Understanding Your Results

After running the analysis, your results will be displayed across four main tabs. Each tab provides a different perspective on how the model is generating responses and how those responses vary across demographic groups. Below is a breakdown of each view and what insights it provides.

---

#### LLM Concepts Tab

This tab extracts high-level concepts directly from the model’s own interpretations of the generated responses. These concepts are grouped into semantic clusters and ranked by frequency.

- The upper chart shows the most frequent concepts across all responses.
- A secondary chart breaks down concept frequency by demographic group.

**Use this tab to:**
- Identify dominant themes or ideas across all responses.
- Understand how certain concepts vary by gender, age, ethnicity, or socioeconomic status.
- Detect potential outliers or unexpected emphasis placed on certain topics for specific groups.

---

#### LDA Concepts Tab

This tab applies Latent Dirichlet Allocation (LDA) to statistically infer topics based on word co-occurrence patterns in the response text.

- Each topic is presented as a list of its top associated keywords.
- Visualizations show topic proportions across the dataset and per demographic group.

**Use this tab to:**
- Reveal hidden patterns in the language that may not be captured by surface-level concept extraction
- Compare how frequently different demographic groups are associated with certain statistically derived topics
- Investigate whether any topics are disproportionately associated with identity-based language or sensitive terms

---

#### Embeddings Tab

This tab uses sentence embeddings and dimensionality reduction (PCA) to map responses into a 2D space, followed by clustering with K-Means.

- Each point in the scatter plot represents a model response
- The points are colored by demographic group to reveal potential clustering by demographics
- Cluster composition is summarized below the chart

**Use this tab to:**
- Explore how responses are grouped based on semantic similarity.
- Detect clusters with uneven demographic representation, which may suggest biased response patterns.

---

#### Agreement Scores Tab

This tab evaluates how consistent the clustering results are across the three methods: LLM-based concepts, LDA topics, and embedding-based clusters.

- A matrix shows agreement scores between each pair of methods using the Hungarian Algorithm.
- A scatter plot visualizes the agreement strength per response.
- A heatmap highlights where clustering methods diverge in their assignments.

**Use this tab to:**
- Assess the reliability and stability of extracted patterns across different analytical techniques.
- Investigate responses with low agreement to better understand ambiguous or inconsistent outputs.
- Determine which clustering method best aligns with your analysis goals or domain expertise.

---

### Conclusion

The Bias Probing Tool is designed to offer a multi-layered understanding of language model behavior across demographic dimensions. Each analytical view (concept extraction, topic modeling, semantic clustering, and agreement scoring) provides a different lens for interpreting model outputs.

By examining both the content and the structure of responses, researchers can uncover:

- Disparities in theme prevalence across demographic groups
- Recurring or dominant model behaviors
- Evidence of implicit bias or preferential framing

This tool provides a robust framework for auditing bias in language model outputs.