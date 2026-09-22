"""
SkillForge — Pre-built skills for Phase 1.

These are the starter skills the system ships with. Each defines a
reusable LangGraph workflow with prompt templates and parameter slots.
"""

from app.models import (
    Skill, SkillParameter, SkillWorkflow, SkillWorkflowStep,
    SkillExample, SkillStats, SkillSource,
)


def get_default_skills() -> list[Skill]:
    return [
        research_topic_skill(),
        compare_tools_skill(),
        summarize_content_skill(),
        find_alternatives_skill(),
    ]


def research_topic_skill() -> Skill:
    return Skill(
        id="skill_research",
        name="research_topic",
        description=(
            "Research a topic thoroughly using web search, "
            "extract key findings from multiple sources, and compile "
            "a structured report with citations."
        ),
        parameters={
            "topic": SkillParameter(
                type="string", description="The topic to research"
            ),
            "depth": SkillParameter(
                type="string",
                description="How deep to go",
                enum=["brief", "detailed", "comprehensive"],
                default="detailed",
            ),
            "num_sources": SkillParameter(
                type="integer",
                description="Number of sources to consult",
                default=5,
            ),
        },
        workflow=SkillWorkflow(
            steps=[
                SkillWorkflowStep(
                    agent="planner",
                    action="decompose_research",
                    description="Break the topic into specific research questions",
                    prompt_template=(
                        "You are a research planner. The user wants to research: {topic}\n"
                        "Depth level: {depth}\n\n"
                        "Break this into 3-5 specific research questions that, when answered, "
                        "would give a thorough understanding of the topic. "
                        "Return ONLY a JSON array of question strings."
                    ),
                ),
                SkillWorkflowStep(
                    agent="researcher",
                    action="web_search",
                    description="Search the web for each research question",
                    prompt_template=(
                        "Search for information about: {query}\n"
                        "Find {num_sources} relevant, high-quality sources.\n"
                        "For each source, extract the key facts and insights.\n"
                        "Return a JSON object with 'sources' array, each having "
                        "'title', 'url', 'key_points' (array of strings)."
                    ),
                ),
                SkillWorkflowStep(
                    agent="formatter",
                    action="compile_report",
                    description="Compile findings into a structured report",
                    prompt_template=(
                        "You are a research report writer. Compile these findings "
                        "into a clear, well-structured report.\n\n"
                        "Topic: {topic}\n"
                        "Research findings:\n{findings}\n\n"
                        "Format as a report with:\n"
                        "- Executive summary (2-3 sentences)\n"
                        "- Key findings (organized by subtopic)\n"
                        "- Notable insights\n"
                        "- Sources used\n"
                        "Write in a clear, professional tone."
                    ),
                ),
            ]
        ),
        examples=[
            SkillExample(
                input="Research transformer architecture in depth",
                output_summary="Comprehensive report on transformer architecture covering self-attention, positional encoding, and recent variants.",
            ),
            SkillExample(
                input="Research the latest developments in agentic AI",
                output_summary="Report on agentic AI covering frameworks, architectures, and industry adoption trends.",
            ),
            SkillExample(
                input="Research how RAG systems work",
                output_summary="Technical overview of retrieval-augmented generation including chunking strategies, embedding models, and reranking.",
            ),
        ],
        stats=SkillStats(),
        source=SkillSource.HARDCODED,
    )


def compare_tools_skill() -> Skill:
    return Skill(
        id="skill_compare",
        name="compare_tools",
        description=(
            "Compare multiple tools, frameworks, or technologies side by side. "
            "Research each one, find their strengths, weaknesses, pricing, "
            "and use cases, then produce a structured comparison."
        ),
        parameters={
            "category": SkillParameter(
                type="string", description="Category of tools to compare (e.g., 'vector databases')"
            ),
            "items": SkillParameter(
                type="string",
                description="Specific tools to compare, comma-separated. If empty, agent finds the top ones.",
                default="",
            ),
            "num_items": SkillParameter(
                type="integer", description="How many tools to compare", default=4
            ),
            "criteria": SkillParameter(
                type="string",
                description="Comparison criteria, comma-separated",
                default="features,pricing,ease of use,community,performance",
            ),
        },
        workflow=SkillWorkflow(
            steps=[
                SkillWorkflowStep(
                    agent="planner",
                    action="identify_items",
                    description="Identify what tools to compare and the comparison criteria",
                    prompt_template=(
                        "The user wants to compare tools in the category: {category}\n"
                        "Specific tools mentioned: {items}\n"
                        "Number of tools to compare: {num_items}\n"
                        "Comparison criteria: {criteria}\n\n"
                        "If specific tools are given, use those. Otherwise, identify the "
                        "top {num_items} most relevant tools in this category.\n"
                        "Return a JSON object with 'tools' (array of tool names) and "
                        "'criteria' (array of comparison dimensions)."
                    ),
                ),
                SkillWorkflowStep(
                    agent="researcher",
                    action="research_each_tool",
                    description="Research each tool individually",
                    prompt_template=(
                        "Research the tool: {tool_name}\n"
                        "Category: {category}\n"
                        "Gather information on these criteria: {criteria}\n\n"
                        "Return a JSON object with the tool name and a 'scores' object "
                        "mapping each criterion to a brief assessment."
                    ),
                ),
                SkillWorkflowStep(
                    agent="formatter",
                    action="build_comparison",
                    description="Build a structured comparison table and recommendation",
                    prompt_template=(
                        "Create a comparison of these tools:\n{tool_data}\n\n"
                        "Format as:\n"
                        "- Overview (one-liner per tool)\n"
                        "- Comparison table (tools as columns, criteria as rows)\n"
                        "- Best for [use case] recommendation for each tool\n"
                        "- Overall verdict\n"
                        "Be specific and opinionated, not wishy-washy."
                    ),
                ),
            ]
        ),
        examples=[
            SkillExample(
                input="Compare the top 5 RAG frameworks",
                output_summary="Side-by-side comparison of LangChain, LlamaIndex, Haystack, etc.",
            ),
            SkillExample(
                input="Compare vector databases: Pinecone vs Weaviate vs Chroma",
                output_summary="Detailed comparison across pricing, performance, ease of use.",
            ),
            SkillExample(
                input="Compare React vs Vue vs Svelte for a new project",
                output_summary="Framework comparison covering DX, performance, ecosystem, hiring.",
            ),
        ],
        stats=SkillStats(),
        source=SkillSource.HARDCODED,
    )


def summarize_content_skill() -> Skill:
    return Skill(
        id="skill_summarize",
        name="summarize_content",
        description=(
            "Summarize content from a URL, topic, or provided text. "
            "Extract the key points, main arguments, and actionable takeaways."
        ),
        parameters={
            "content": SkillParameter(
                type="string", description="URL, topic name, or raw text to summarize"
            ),
            "format": SkillParameter(
                type="string",
                description="Output format",
                enum=["bullet_points", "paragraph", "executive_summary", "eli5"],
                default="bullet_points",
            ),
            "max_points": SkillParameter(
                type="integer", description="Max key points to extract", default=10
            ),
        },
        workflow=SkillWorkflow(
            steps=[
                SkillWorkflowStep(
                    agent="researcher",
                    action="fetch_content",
                    description="Fetch or search for the content to summarize",
                    prompt_template=(
                        "The user wants a summary of: {content}\n"
                        "If this is a URL, fetch and extract the main content.\n"
                        "If this is a topic, search for the most authoritative overview.\n"
                        "Return the full text content to be summarized."
                    ),
                ),
                SkillWorkflowStep(
                    agent="formatter",
                    action="summarize",
                    description="Create a structured summary",
                    prompt_template=(
                        "Summarize the following content:\n{fetched_content}\n\n"
                        "Format: {format}\n"
                        "Maximum key points: {max_points}\n\n"
                        "Focus on:\n"
                        "- The core argument or main point\n"
                        "- Key supporting evidence or findings\n"
                        "- Actionable takeaways\n"
                        "- What's novel or surprising\n"
                        "Skip filler and obvious context."
                    ),
                ),
            ]
        ),
        examples=[
            SkillExample(
                input="Summarize the LangGraph documentation",
                output_summary="Key points about LangGraph's graph-based agent orchestration.",
            ),
            SkillExample(
                input="Summarize recent news about OpenAI",
                output_summary="Latest developments at OpenAI including model releases and strategy.",
            ),
        ],
        stats=SkillStats(),
        source=SkillSource.HARDCODED,
    )


def find_alternatives_skill() -> Skill:
    return Skill(
        id="skill_alternatives",
        name="find_alternatives",
        description=(
            "Find alternatives to a specific tool, service, or technology. "
            "Search for competitors, evaluate their strengths, and rank by relevance."
        ),
        parameters={
            "tool": SkillParameter(
                type="string", description="The tool/service to find alternatives for"
            ),
            "reason": SkillParameter(
                type="string",
                description="Why the user wants alternatives (cost, features, etc.)",
                default="general",
            ),
            "num_alternatives": SkillParameter(
                type="integer", description="How many alternatives to find", default=5
            ),
        },
        workflow=SkillWorkflow(
            steps=[
                SkillWorkflowStep(
                    agent="researcher",
                    action="search_alternatives",
                    description="Search for alternatives to the specified tool",
                    prompt_template=(
                        "Find the top {num_alternatives} alternatives to: {tool}\n"
                        "User's reason for switching: {reason}\n\n"
                        "Search for competitors and alternatives. For each, find:\n"
                        "- Name and one-line description\n"
                        "- Key differentiator vs {tool}\n"
                        "- Pricing (if applicable)\n"
                        "- Pros and cons\n"
                        "Return as a JSON array of alternatives."
                    ),
                ),
                SkillWorkflowStep(
                    agent="formatter",
                    action="rank_and_format",
                    description="Rank alternatives and format the output",
                    prompt_template=(
                        "Rank and format these alternatives to {tool}:\n{alternatives}\n\n"
                        "User's switching reason: {reason}\n\n"
                        "Format as:\n"
                        "- Ranked list with #1 being the best match for their reason\n"
                        "- One paragraph per alternative explaining why it's good/bad\n"
                        "- A 'Quick pick' recommendation at the top\n"
                        "Be opinionated. Don't say they're all great."
                    ),
                ),
            ]
        ),
        examples=[
            SkillExample(
                input="Find free alternatives to Pinecone for vector search",
                output_summary="Ranked list including Chroma, Weaviate, Milvus with pricing focus.",
            ),
            SkillExample(
                input="Find alternatives to Vercel for deploying React apps",
                output_summary="Comparison of Netlify, Railway, Cloudflare Pages, Render.",
            ),
        ],
        stats=SkillStats(),
        source=SkillSource.HARDCODED,
    )
