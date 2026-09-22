from pathlib import Path

from deep_reporter.api import create_openai_api


# Model Parameter Settings
api = create_openai_api({
    "primary_model": "gpt-5-mini",
    "retriever_url": "http://localhost:5555/search",
    # other keys (api keys, base urls, vlm config) default to the env vars
    # exported in Step 4; pass them here to override per-call.
})


result = api.generate_article(
    # Input your topic
    overall_query="Analyze the impact of large language models on scientific research",
    overall_checklist=[
        "Cover key application areas across disciplines",
        "Discuss methodological shifts in literature review and writing",
        "Include limitations and risks",
    ],
    generation_mode="with_planner",
    text_topk=20,
    image_topk=10,
    enable_filter=False,  # Whether to enable VLM checking
    userid="deconstruct_AG001",  # used as the eval-side join key
)


if result["success"]:
    article = result["final_article"]

    output_path = Path(__file__).resolve().parent / "result.md"
    output_path.write_text(article, encoding="utf-8")

    print(f"Report generated successfully: {output_path}")
else:
    print(f"Report generation failed: {result['error']}")