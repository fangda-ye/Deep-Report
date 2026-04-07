# example_usage.py
"""
Usage examples for the LongGen multimodal long-form generation system.
"""
from deep_reporter.api import create_openai_api

def example_with_planner():
    """Use automatic outline generation mode."""
    config = {
        "openai_api_key": "your-api-key",
        "openai_base_url": "https://api.openai.com/v1",
        "primary_model": "gpt-5-mini",
        "auxiliary_model": "gpt-5-mini",
        "retriever_url": "http://localhost:5555/search",
        "log_dir": "logs",
        "log_file_prefix": "example_with_planner"
    }
    
    api = create_openai_api(config)

    result = api.generate_article(
        overall_query="Explain the transformer architecture in deep learning",
        overall_checklist=[
            "Explain the attention mechanism",
            "Describe the encoder-decoder structure",
            "Discuss key innovations and applications",
            "Include relevant diagrams and visualizations"
        ],
        generation_mode="with_planner",
        text_topk=20,
        image_topk=10,
        enable_filter=True,
        userid="example_user"
    )
    
    if result["success"]:
        print("Article generated successfully!")
        print(f"Total sections: {result['total_sections']}")
        print(f"Article length: {len(result['final_article'])} chars")
        print(f"\n--- Generated Article ---\n{result['final_article'][:500]}...")

        print(f"\n--- Generated Outline ---")
        for idx, section in enumerate(result['outline']):
            print(f"Section {idx+1}: {section['section_description']}")
    else:
        print(f"Generation failed: {result['error']}")


def example_without_planner():
    """Use a user-provided outline."""
    config = {
        "openai_api_key": "your-api-key",
        "openai_base_url": "https://api.openai.com/v1",
        "primary_model": "gpt-5-mini",
        "retriever_url": "http://localhost:5555/search"
    }
    
    api = create_openai_api(config)

    # User-provided outline
    custom_outline = [
        {
            "section_description": "Introduction to Transformer Architecture",
            "sectional_checklist": [
                "Brief history of sequence models",
                "Motivation for transformers",
                "Overview of key components"
            ]
        },
        {
            "section_description": "Attention Mechanism Deep Dive",
            "sectional_checklist": [
                "Self-attention concept",
                "Multi-head attention",
                "Mathematical formulation",
                "Include attention visualization diagram"
            ]
        },
        {
            "section_description": "Encoder and Decoder Structures",
            "sectional_checklist": [
                "Encoder stack details",
                "Decoder stack details",
                "Position encoding",
                "Include architecture diagram"
            ]
        },
        {
            "section_description": "Applications and Impact",
            "sectional_checklist": [
                "NLP applications (BERT, GPT)",
                "Vision transformers",
                "Recent developments",
                "Future directions"
            ]
        }
    ]
    
    result = api.generate_article(
        overall_query="Comprehensive guide to Transformer architecture",
        overall_checklist=[
            "Cover all major components",
            "Include technical details and math",
            "Provide visual aids",
            "Discuss real-world applications"
        ],
        generation_mode="without_planner",
        outline=custom_outline,
        text_topk=15,
        image_topk=8,
        enable_filter=False,
        userid="example_user"
    )
    
    if result["success"]:
        print("Article generated successfully!")
        print(f"Total sections: {result['total_sections']}")
        print(f"Article length: {len(result['final_article'])} chars")
        print(f"\n--- Generated Article ---\n{result['final_article'][:500]}...")
    else:
        print(f"Generation failed: {result['error']}")


def example_from_blueprint():
    """Build training data from a provided blueprint."""
    
    config = {
        "deepseek_api_key": "your-deepseek-key",
        "deepseek_base_url": "https://api.deepseek.com/v1",
        "primary_model": "deepseek-r1",
        "auxiliary_model": "qwen-plus-latest",
        "retriever_url": "http://localhost:7000/search"
    }
    
    api = create_openai_api(config)
    
    # Simulated blog deconstruction result
    blog_outline = [
        {
            "section_description": "Introduction to the topic",
            "sectional_checklist": [
                "Define key terms",
                "State the main problem",
                "Preview the article structure"
            ]
        },
        {
            "section_description": "Background and related work",
            "sectional_checklist": [
                "Review existing approaches",
                "Identify limitations",
                "Motivate the proposed solution"
            ]
        },
        {
            "section_description": "Methodology and approach",
            "sectional_checklist": [
                "Describe the core method",
                "Provide technical details",
                "Include relevant diagrams"
            ]
        },
        {
            "section_description": "Results and evaluation",
            "sectional_checklist": [
                "Present experimental results",
                "Compare with baselines",
                "Discuss findings"
            ]
        },
        {
            "section_description": "Conclusion and future work",
            "sectional_checklist": [
                "Summarize key contributions",
                "Discuss limitations",
                "Suggest future directions"
            ]
        }
    ]
    
    result = api.generate_article(
        overall_query="Novel approach to multimodal learning",
        overall_checklist=[
            "Technical depth and accuracy",
            "Clear explanations with examples",
            "Visual aids where appropriate",
            "Comprehensive coverage of the topic"
        ],
        generation_mode="without_planner",
        outline=blog_outline,
        text_topk=20,
        image_topk=10,
        enable_filter=True,
        userid="training_data_gen"
    )
    
    if result["success"]:
        print("Training sample generated successfully!")
        training_sample = {
            "overall_query": "Novel approach to multimodal learning",
            "outline": blog_outline,
            "generated_sections": []
        }
        
        for section_info in result['outline']:
            if section_info.get('content'):
                training_sample["generated_sections"].append({
                    "section_description": section_info["section_description"],
                    "sectional_checklist": section_info["sectional_checklist"],
                    "retrieved_texts": section_info.get("retrieved_texts", []),
                    "retrieved_images": section_info.get("retrieved_images", []),
                    "generated_content": section_info["content"]
                })
        
        print(f"Generated {len(training_sample['generated_sections'])} section-level training samples")
        import json
        with open("training_sample.json", "w") as f:
            json.dump(training_sample, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print("=" * 60)
    print("LongGen Usage Examples")
    print("=" * 60)

    print("\nUncomment the desired example function to run it.")