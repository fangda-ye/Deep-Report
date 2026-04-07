"""
Evaluation prompt templates for article assessment.
"""

SECTION_AND_ARTICLE_EVAL_TEMPLATE = """You are a strict, critical expert evaluator for long-form generated articles. Your task is to evaluate both individual sections and the overall article quality using a simple 3-level scoring system.

## Scoring System: 3-Level Scale (1, 3, 5)

**IMPORTANT**: You must use ONLY these 3 values: **1, 3, 5**. No other values are allowed.

### Score Definitions

**5 - Good (Genuinely Impressive)**
- Comprehensive coverage with real depth and insight
- Goes beyond surface-level treatment with concrete examples and expert-level analysis
- Would genuinely impress a domain expert
- You can point to specific things that are notably well done
- **This should be RARE** - most LLM content does NOT deserve a 5

**3 - Acceptable (Gets the Job Done)**
- Covers the required points adequately
- Correct but generic/surface-level treatment
- Meets basic requirements but nothing stands out
- Typical LLM-generated quality: fluent but not deep
- **This is the DEFAULT** - when in doubt, give 3

**1 - Poor (Has Problems)**
- Missing important aspects or contains errors
- Superficial, rushed, or poorly organized
- Fails to adequately address the requirements
- Would leave a reader with significant questions

### ⚠️ CRITICAL Scoring Rules

1. **Default to 3**: Most LLM-generated content is "acceptable but not impressive". If you're unsure, give 3.

2. **5 is RARE**: Do NOT give 5 just because content is "good" or "correct". Ask yourself:
   - Does this genuinely stand out as exceptional?
   - Would an expert be impressed, or just satisfied?
   - Can I point to something specific that goes beyond expectations?
   - If answers are "no", give 3, not 5.

3. **1 is for real problems**: Missing content, errors, or poor quality. Don't hesitate to use it.

4. **LLM content often looks better than it is**: Fluent writing ≠ deep content. Look past the polish.

---

## Article Overview
**Query**: {query}
**Overall Checklist**: {overall_checklist}
**Number of Sections**: {num_sections}

---

## Evaluation Tasks

### Part 1: Section-Level Evaluation

For each section, evaluate:

#### 1.1 Description Completion Score (1/3/5)
Assess how well the generated content fulfills the intended section description.

**Scoring:**
- **5**: Comprehensive coverage with excellent depth, concrete examples, expert insight. Rare.
- **3**: Covers the main points adequately but surface-level or generic. Default.
- **1**: Major gaps, errors, or shallow treatment that fails requirements.

#### 1.2 Checklist Item Evaluation (1/3/5 per item)
For each checklist item, evaluate how well it is addressed.

**Scoring:**
- **5**: Thoroughly addressed with depth, examples, and insight. Genuinely impressive. Rare.
- **3**: Mentioned and basically covered, but generic or surface-level. Default.
- **1**: Missing, barely touched, or incorrect.

---

### Part 2: Article-Level Evaluation

Evaluate the entire article on these dimensions (each scored 1/3/5):

#### 2.1 Coherence
**What to assess:**
- Logical flow between sections
- Transition quality between topics
- Consistency of perspective
- Citation & image format compliance:
  - Text citations: `[citation:txt1]`, `[citation:txt2]`, etc.
  - Image insertions: `![](citation:img1)`, `![](citation:img2)`, etc.

**Scoring:**
- **5**: Excellent flow, smooth transitions, proper formatting. Reads as unified whole.
- **3**: Generally coherent with some awkward transitions or minor format issues.
- **1**: Disjointed, confusing structure, or major formatting problems.

#### 2.2 Fluency
**What to assess:**
- Sentence construction and variety
- Vocabulary appropriateness
- Readability and grammar

**Scoring:**
- **5**: Professional, polished, engaging writing throughout.
- **3**: Clear and readable but unremarkable. Typical LLM quality.
- **1**: Awkward, difficult to read, or has errors.

#### 2.3 Repetition (Higher = Less Repetition)
**What to assess:**
- Cross-section redundancy
- Phrase/sentence repetition
- Unnecessary re-explanation

**Scoring:**
- **5**: No noticeable repetition. Each section adds new value.
- **3**: Some repetition but not severely distracting.
- **1**: Significant redundancy that hurts quality.

#### 2.4 Termination (Higher = Better)
**What to assess:**
- Inappropriate conclusions in non-final sections (e.g., "In conclusion..." mid-article)
- Premature summary statements

**Scoring:**
- **5**: No inappropriate concluding statements. Clean section boundaries.
- **3**: Minor issues with section endings.
- **1**: Frequent inappropriate termination patterns.

---

## Sections Data

{sections_data}

---

## Output Format

Respond with a valid JSON object (no markdown code blocks, just raw JSON) with this exact structure:

{{
  "section_evaluations": [
    {{
      "section_index": 0,
      "description_completion_score": <must be 1, 3, or 5>,
      "description_completion_reasoning": "Explain why this score was chosen",
      "checklist_evaluations": [
        {{
          "checklist_item": "The exact checklist item text",
          "score": <must be 1, 3, or 5>,
          "reasoning": "Brief explanation"
        }}
      ]
    }}
  ],
  "article_evaluation": {{
    "coherence_score": <must be 1, 3, or 5>,
    "coherence_reasoning": "Explain the logical flow and transition quality",
    "fluency_score": <must be 1, 3, or 5>,
    "fluency_reasoning": "Explain writing quality and readability",
    "repetition_score": <must be 1, 3, or 5>,
    "repetition_reasoning": "Explain level of redundancy (higher = less repetition)",
    "termination_score": <must be 1, 3, or 5>,
    "termination_reasoning": "Explain appropriateness of section endings (higher = better)"
  }}
}}

**CRITICAL**: All scores must be exactly one of: 1, 3, 5. No other values are allowed. When in doubt, give 3.
"""


# Template for text citation evaluation
TEXT_CITATION_EVAL_TEMPLATE = """You are an expert evaluator for citation relevance. Your task is to determine if a text citation is relevant and properly used in context.

## Citation Context
**Section Description**: {section_description}
**Context Before Citation**: {context_before}
**Citation Marker**: {citation_marker}
**Context After Citation**: {context_after}

## Source Content
{source_content}

## Evaluation Task
Evaluate the relevance of this citation based on:
1. **Semantic Relevance**: Does the source content support or relate to the claims made in the context?
2. **Usage Appropriateness**: Is the citation used in a way that accurately represents the source?
3. **Contextual Fit**: Does the citation fit naturally within the surrounding text?

## Output Format
Respond with a valid JSON object (no markdown code blocks) with this structure:

{{
  "relevance_score": 8.5,
  "semantic_relevance": 9.0,
  "usage_appropriateness": 8.0,
  "contextual_fit": 8.5,
  "reasoning": "Brief explanation of the scores",
  "issues": ["List any issues found, empty array if none"]
}}

All scores should be numbers between 0 and 10, where 10 is perfectly relevant and 0 is completely irrelevant.
The overall relevance_score should be the average or weighted combination of the three sub-scores.
"""


# Template for section-level image-text coherence evaluation
SECTION_IMAGE_TEXT_EVAL_TEMPLATE = """You are a strict, meticulous, and objective expert evaluator for multimodal content in academic/technical writing. Your task is to evaluate the quality of image usage in a section of an article.

## Core Scoring Rules (Apply to ALL scores)
1. **Use a scale of 0-10 (continuous values)**: Do not cluster scores around 8-10.
   - **8-10 points**: Excellent/outstanding performance. Fully meets or exceeds the criterion requirements.
   - **6-8 points**: Good performance. Largely meets the criterion requirements with notable strengths.
   - **4-6 points**: Average performance. Basically meets the criterion requirements, neither good nor bad.
   - **2-4 points**: Poor performance. Minimally meets the criterion requirements with significant deficiencies.
   - **0-2 points**: Very poor performance. Almost completely failed or missing.
2. **Be Harsh**: Default to a lower score if you are unsure. Penalize inappropriate or poorly integrated images heavily.
3. **Evaluate All Dimensions Independently**: Each dimension measures a different aspect of image usage quality.
4. **IMPORTANT - Always Return Numeric Scores**: Never use "N/A", "None", or text descriptions as scores. Always provide a numeric value (0-10) based on the evaluation criteria below.

## Section Context
**Section Description**: {section_description}

**Section Content (with image placeholders)**:
{section_content}

**Number of Images in Section**: {num_images}

## Images to Evaluate
{images_info}

**CRITICAL INSTRUCTION**:
- You MUST carefully examine the ACTUAL IMAGES provided below (not text descriptions)
- Base your evaluation on what you directly observe in the visual content
- Analyze the visual elements, layout, clarity, and relevance that you see
- Do not rely on or assume content based on text descriptions or expectations

## Evaluation Task

Evaluate the section's image usage based on the following dimensions (each scored 0-10):

### 1. Image Richness (richness_score)
Evaluate whether the quantity and variety of images are appropriate:
- Is the number of images suitable for the section length and complexity?
- Are images distributed well throughout the section (not clustered)?
- Does the variety of images (diagrams, charts, photos, etc.) match the content needs?

**Special Case - If there are NO images (num_images = 0):**
- Analyze the section content to determine if images are necessary
- **If images are NOT needed** (e.g., purely conceptual discussion, definitions, abstract theory):
  - Score: 7-10 points (no images needed, appropriate decision)
- **If images ARE needed** (e.g., describing architectures, methods, processes, experimental results):
  - Score: 0-3 points (missing essential visual aids)
- **Uncertain cases** (e.g., could benefit from images but not strictly necessary):
  - Score: 4-6 points (adequate but could be improved)

### 2. Image-Text Coherence (coherence_score)
Evaluate how well images relate to and support the text content:
- Do all images directly relate to and support the text content?
- Are the visual elements aligned with what the text is explaining?
- Does each image add meaningful information that complements the text?
- Are there any irrelevant or tangentially related images?

**Special Case - If there are NO images:**
- If images are not needed: Score 8-10 (no coherence issues, appropriate)
- If images are needed: Score 0-3 (no support for text, poor coherence)
- If uncertain: Score 5 (neutral, neither good nor bad)

### 3. Placement & Integration (placement_score)
Evaluate the positioning and integration of images within the text flow:
- Are images placed at logical points in the text flow?
- Does the text provide proper context before/after each image?
- Are transitions between text and images smooth and natural?
- Do images appear when relevant concepts are discussed (not too early or late)?

**Special Case - If there are NO images:**
- If images are not needed: Score 8-10 (no placement issues, appropriate)
- If images are needed: Score 0-3 (no integration, missing placement)
- If uncertain: Score 5 (neutral)

### 4. Visual Quality & Clarity (clarity_score)
Evaluate the quality and understandability of the images themselves:
- Are the images clear and easy to understand?
- Are key elements in each image visible and discernible?
- Are images appropriate for the academic/technical context?
- Do images maintain consistent quality and style?

**Special Case - If there are NO images:**
- If images are not needed: Score 8-10 (no quality issues, appropriate)
- If images are needed: Score 0-3 (no visual quality, missing)
- If uncertain: Score 5 (neutral)

## Content Types That Typically NEED Images

Consider the following content types as typically requiring visual aids:
1. **System/Model Architecture**: Diagrams showing components and connections
2. **Algorithms/Methods**: Flowcharts or pseudocode visualizations
3. **Experimental Setup**: Photos or diagrams of equipment/environments
4. **Results/Data**: Charts, graphs, tables showing findings
5. **Processes/Workflows**: Step-by-step visual representations
6. **Comparisons**: Side-by-side visual comparisons
7. **Examples/Case Studies**: Concrete visual examples

## Content Types That May NOT Need Images

1. **Abstract Definitions**: Pure conceptual explanations
2. **Literature Review**: Discussion of related work (unless comparing approaches)
3. **Theoretical Background**: Mathematical proofs, theoretical foundations
4. **Introductory Text**: Background context, motivation
5. **Conclusions**: Summary statements, future work discussions

## Output Format

You MUST respond with ONLY a valid JSON object. Follow this EXACT structure:

{{
  "richness_score": <float between 0-10>,
  "richness_reasoning": "<your explanation here>",
  "coherence_score": <float between 0-10>,
  "coherence_reasoning": "<your explanation here>",
  "placement_score": <float between 0-10>,
  "placement_reasoning": "<your explanation here>",
  "clarity_score": <float between 0-10>,
  "clarity_reasoning": "<your explanation here>"
}}

**CRITICAL OUTPUT REQUIREMENTS - READ CAREFULLY:**

1. **Score Fields Must Contain NUMBERS ONLY** (not text):
   - Replace `<float between 0-10>` with an actual number like 7.5 or 3.2
   - ✅ CORRECT: "richness_score": 7.5
   - ✅ CORRECT: "richness_score": 3.0
   - ❌ WRONG: "richness_score": "This section lacks images"
   - ❌ WRONG: "richness_score": "N/A"
   - ❌ WRONG: "richness_score": "Since no images are present..."
   - ❌ WRONG: "richness_score": "Continuous score 0-10"

2. **Reasoning Fields Must Contain TEXT ONLY**:
   - Replace `<your explanation here>` with your actual explanation
   - ✅ CORRECT: "richness_reasoning": "Brief explanation of the score"
   - ❌ WRONG: Put explanations in the score field

3. **For sections with NO images**:
   - Still provide NUMERIC scores (not text or "N/A")
   - Low score if images are needed
   - High score if images not needed
   - Put explanation in reasoning field
"""
