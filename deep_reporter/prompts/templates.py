# LongGen/prompts/templates.py
"""
Prompt templates for multimodal long-form generation
"""

# Outline Generation Template (used in planner_node)
OUTLINE_GENERATION_TEMPLATE = """You are an expert at generating article outlines.

Given an overall query/topic and requirements checklist, generate a detailed outline for a long-form article.

**Overall Query/Topic:**
{overall_query}

**Overall Requirements Checklist:**
{overall_checklist}

**Task:**
Generate a structured outline with 3-7 sections. For each section, provide:

1. **section_description**: A DETAILED description (2-4 sentences) that clearly explains:
   - What this section is about
   - What specific aspects will be covered
   - How it contributes to the overall article
   - What the reader should learn from this section

   **IMPORTANT**: DO NOT just write a title or short phrase. Write a comprehensive description that provides enough context for a writer to understand what needs to be written.

2. **sectional_checklist**: A list of 3-5 specific, actionable requirements for this section

**Output Format (JSON):**
```json
{{
  "outline": [
    {{
      "section_description": "This section provides a comprehensive introduction to the transformer architecture in deep learning. It begins by explaining the historical context of sequence-to-sequence models and the limitations that transformers aimed to address. The section establishes why transformers represent a paradigm shift in neural network design and previews the key innovations (attention mechanism, positional encoding, parallel processing) that will be explored in detail in subsequent sections. Readers will gain foundational understanding of what transformers are and why they matter.",
      "sectional_checklist": [
        "Define what transformers are and their primary purpose in deep learning",
        "Explain the historical context and limitations of previous sequence models (RNNs, LSTMs)",
        "Preview the key innovations that make transformers effective",
        "Establish the importance and widespread adoption of transformer architectures"
      ]
    }},
    {{
      "section_description": "This section provides an in-depth technical explanation of the attention mechanism, which is the core innovation of transformers. It covers the mathematical foundations of self-attention, including query, key, and value matrices, and explains how attention weights are computed and applied. The section also discusses multi-head attention and how it allows the model to attend to different aspects of the input simultaneously. Readers will understand both the conceptual intuition and mathematical details of how attention works.",
      "sectional_checklist": [
        "Explain the concept of self-attention and why it's powerful",
        "Detail the mathematical computation: Q, K, V matrices and attention scores",
        "Describe multi-head attention and its benefits",
        "Include diagrams showing attention weight visualization"
      ]
    }},
    ...
  ]
}}
```

**Key Guidelines:**
- Each section_description should be 2-4 sentences providing rich context
- Avoid single-sentence or title-only descriptions
- Be specific about what content will be covered and why
- Think about what information a writer would need to create high-quality content

Generate the outline now:
"""

# Query Generation Template (used in query_generation_node)
QUERY_GENERATION_TEMPLATE = """You are an expert at generating search queries for multimodal information retrieval.

Given the article context and current section requirements, generate targeted search queries to find relevant text passages and images.

**Overall Article Query/Topic:**
{overall_query}

**Overall Requirements:**
{overall_checklist}

**Previous Sections Summary:**
{previous_sections_summary}

**Current Section Description:**
{section_description}

**Current Section Requirements:**
{sectional_checklist}

**Task:**
Generate search queries that will help retrieve relevant information for writing this section.
Generate TWO types of queries:
1. **text_queries**: For finding related text passages, explanations, or documentation (3-5 queries)
2. **image_queries**: For finding relevant diagrams, charts, figures, or visualizations (2-4 queries)

**Output Format (JSON):**
```json
{{
  "text_queries": [
    "query 1 for text retrieval",
    "query 2 for text retrieval",
    "query 3 for text retrieval"
  ],
  "image_queries": [
    "query 1 for image retrieval",
    "query 2 for image retrieval"
  ]
}}
```

Generate the queries now:
"""

# Section Writing Template - THE CORE TEMPLATE
SECTION_WRITING_TEMPLATE = """# Role
You are an expert writer skilled at creating coherent, well-structured long-form content with proper source citations.

# Task
Write a specific section of a larger article based on the provided requirements, context, and retrieved materials.

# Overall Article Context
**Article Topic:** {overall_query}

**Overall Requirements:**
{overall_checklist}

# Current Section Information
**Section Position:** {position_info}
**Section Description:** {section_description}

**Section Requirements (must address all):**
{sectional_checklist}

# Context from Previous Sections
**Summary of previous sections:**
{previous_sections_summary}

**End of previous section (for smooth transition):**
{previous_section_tail}

# Retrieved Materials
You have access to the following sources for citation:
{retrieved_materials}

# CRITICAL Citation Rules
You MUST follow these citation rules strictly:

1. **Text Citation Format:**
   - When using information from a text source, cite it as: `[citation:txt1]`, `[citation:txt2]`, etc.
   - Example: "Recent studies show significant progress in AI safety[citation:txt3]."

2. **Image Citation Format:**
   - When you want to insert an image/figure, use: `![](citation:img1)`, `![](citation:img2)`, etc.
   - Example: "The architecture is shown below:\n![](citation:img5)\nAs illustrated, the system..."
   - ONLY use this format for sources explicitly marked as "Image"

3. **Citation Placement:**
   - Place text citations at the end of the sentence or claim
   - Place image citations on their own line where the visual should appear
   - You can cite the same source multiple times if needed

4. **Important Distinctions:**
   - Text sources → use `[citation:txt1]`, `[citation:txt2]`, etc.
   - Image sources → use `![](citation:img1)`, `![](citation:img2)`, etc.

# Position-Aware Writing Instructions

**Is this the FIRST section?** {is_first_section}
**Is this the LAST section?** {is_last_section}

## If FIRST section (is_first_section = yes):
- Start with an engaging introduction
- No need to reference previous content (there is none)
- Set the stage for the article
- DO NOT conclude the entire article

## If MIDDLE section (is_first_section = no, is_last_section = no):
- Begin with a smooth transition from the previous section
- Focus ONLY on this section's specific requirements
- **CRITICAL: DO NOT use concluding phrases like:**
  - "In conclusion"
  - "To summarize"
  - "Overall, this article"
  - "We have discussed"
  - "In this article, we covered"
- Continue the narrative without wrapping up the entire article
- End in a way that allows the next section to continue naturally

## If LAST section (is_last_section = yes):
- Provide appropriate transitions from previous content
- You MAY synthesize and conclude the entire article
- Wrap up all major points discussed in the article

# Content Requirements
1. **Address all sectional requirements** in the checklist above
2. **Maintain coherence** with previous sections using the provided context
3. **Use retrieved materials** as the foundation of your content
4. **Cite sources properly** using the formats specified above
5. **Write naturally** - your content should flow smoothly while integrating citations
6. **Appropriate length** - typically 300-800 words depending on the complexity of requirements

# Markdown Formatting Requirements
**CRITICAL: You MUST use proper Markdown formatting for structure:**

1. **Section Title (Required):**
   - Start your content with a section title using `# Title`
   - The title should reflect the section description

2. **Subsections (If Needed):**
   - Use `## Subsection Title` for major subsections
   - Use `### Subsubsection Title` for deeper nesting
   - Use appropriate heading levels to create clear hierarchy

3. **Example Structure:**
```markdown
# Introduction to Machine Learning

Machine learning has revolutionized...[citation:txt1]

## Supervised Learning
Supervised learning approaches...[citation:txt2]

### Classification
Classification tasks involve...[citation:txt3]

## Unsupervised Learning
Unlike supervised methods...[citation:txt5]
```

# Output Format
Provide ONLY the section content with proper Markdown formatting and citations. Do NOT include:
- Meta-commentary about what you're doing
- Explanations of your writing process

**Remember to start with a section title using `#`!**

Begin writing the section now:"""

# Filter Template (optional, used in filter_node)
FILTER_TEMPLATE = """# Role
You are an efficient information filtering assistant.

# Task
Determine which of the provided information sources are highly relevant to the current section requirements.

# Context
**Current Section Description:** {section_description}

**Section Requirements:**
{sectional_checklist}

**Information Sources to Filter:**
{sources_list}

# Instructions
1. Read each numbered source carefully (txt1, txt2, txt3, etc.)
2. Determine if it provides valuable information for the section requirements
3. Return the numbers of ALL useful sources

# Output Format
Return ONLY a list of numbers (comma-separated or Python list format):
- Correct example 1: 1, 3, 5, 8
- Correct example 2: [1, 3, 5, 8]
- If none are useful: []

Note: The sources are labeled as txt1, txt2, etc., but you should return just the numbers (1, 2, 3...).

Your response:"""

# Image Filter Template (VLM-based filtering)
IMAGE_FILTER_TEMPLATE = """# Task
You are an expert at evaluating image relevance for academic/technical content.

# Context
**Current Section Description:** {section_description}

**Section Requirements:**
{sectional_checklist}

**Image Description:** {image_description}

# Question
Based on the section requirements above, is this image relevant and useful for this section?

Consider:
1. Does the image visually illustrate concepts mentioned in the requirements?
2. Is it a diagram, chart, figure, or visualization that adds value?
3. Does it relate to the section topic?

# Response Format
Answer with ONLY ONE WORD:
- "YES" if the image is relevant and should be kept
- "NO" if the image is not relevant

Your answer:"""