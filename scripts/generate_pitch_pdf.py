"""
Generates the complete, beautifully styled ApexRAG Interview Mastery & Pitch Guide PDF.
Designed for candidates needing plain-English explanations, full forms, anxiety tips,
and word-for-word interview scripts.
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


def generate_pitch_pdf(output_path: str = "reports/ApexRAG_Interview_Mastery_Guide.pdf"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom Palettes
    navy = colors.HexColor("#1E293B")
    teal = colors.HexColor("#0F766E")
    teal_light = colors.HexColor("#CCFBF1")
    charcoal = colors.HexColor("#334155")
    bg_light = colors.HexColor("#F8FAFC")
    border_color = colors.HexColor("#CBD5E1")
    amber = colors.HexColor("#D97706")
    green = colors.HexColor("#16A34A")

    # Typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=navy,
        alignment=1, # Center
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=teal,
        alignment=1,
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=navy,
        spaceBefore=14,
        spaceAfter=6,
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=teal,
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=charcoal,
        spaceAfter=5,
    )
    body_bold = ParagraphStyle(
        "BodyBoldCustom",
        parent=body_style,
        fontName="Helvetica-Bold",
    )
    script_style = ParagraphStyle(
        "ScriptSpeech",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9.5,
        leading=14,
        textColor=navy,
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=charcoal,
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11.5,
        textColor=navy,
    )

    story = []

    # ========================== COVER / HEADER ==========================
    story.append(Paragraph("⚡ ApexRAG: Complete Interview Mastery Guide", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Plain-English Pitch, Confidence Strategies, Full Forms & Word-for-Word Cross-Questions", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2, color=teal, spaceBefore=4, spaceAfter=14))

    # ========================== SECTION 1: CONFIDENCE & NERVOUSNESS ==========================
    story.append(Paragraph("1. How to Handle Nervousness, Fear & Hand-Shaking", h1_style))
    story.append(Paragraph(
        "It is completely normal to feel nervous, have shaky hands, or worry about your English. "
        "Remember this secret: <b>Interviewers do NOT judge your English accent; they judge whether you can solve technical problems.</b> "
        "Here are 4 simple physical rules to use before and during your interview:",
        body_style,
    ))

    tips_data = [
        [
            Paragraph("<b>Rule 1: Slow Down Your Speech</b>", table_cell_bold),
            Paragraph("When people are scared, they speak too fast. Consciously speak <b>30% slower</b>. Slow speech makes you sound confident, calm, and senior. It also gives your brain time to think of the next word.", table_cell),
        ],
        [
            Paragraph("<b>Rule 2: The 3-Second Pause</b>", table_cell_bold),
            Paragraph("Never answer immediately. When the interviewer finishes asking, take a deep breath, wait 3 full seconds, and say: <i>'That is a great question. Let me break it down into two parts.'</i> Pausing stops hand shaking instantly.", table_cell),
        ],
        [
            Paragraph("<b>Rule 3: What to Say If Your Mind Goes Blank</b>", table_cell_bold),
            Paragraph("If you freeze or forget a word, <b>do not panic</b>. Smile and say: <i>'Excuse me, my mind just paused for a second. Let me rephrase that.'</i> Interviewers respect this honesty 100 times more than pretending.", table_cell),
        ],
        [
            Paragraph("<b>Rule 4: Physical Hand Grounding</b>", table_cell_bold),
            Paragraph("Keep your hands resting flat on your desk or knees under the camera. Do not hold pens or paper that shake. Take 3 deep diaphragmatic breaths through your nose before entering the call.", table_cell),
        ],
    ]
    t_tips = Table(tips_data, colWidths=[1.8 * inch, 5.2 * inch])
    t_tips.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_light),
        ('BOX', (0, 0), (-1, -1), 1, border_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_tips)
    story.append(Spacer(1, 12))

    # ========================== SECTION 2: GLOSSARY OF ALL FULL FORMS ==========================
    story.append(Paragraph("2. The Plain-English Glossary: Every Full Form Spelled Out", h1_style))
    story.append(Paragraph("Use these full forms to explain terms easily to anyone, even if they have zero technical background:", body_style))

    glossary_data = [
        [Paragraph("<b>Term</b>", table_cell_bold), Paragraph("<b>Full Form</b>", table_cell_bold), Paragraph("<b>Simple Meaning (Like Explaining to a 10-Year-Old)</b>", table_cell_bold)],
        [
            Paragraph("<b>RAG</b>", table_cell_bold),
            Paragraph("Retrieval-Augmented Generation", table_cell),
            Paragraph("Instead of letting the AI guess from memory (which causes lies), you give it an open textbook to read before answering.", table_cell),
        ],
        [
            Paragraph("<b>AST</b>", table_cell_bold),
            Paragraph("Abstract Syntax Tree", table_cell),
            Paragraph("A smart scanner that understands programming code so computer code blocks are never cut in half when splitting text.", table_cell),
        ],
        [
            Paragraph("<b>BM25</b>", table_cell_bold),
            Paragraph("Best Matching 25", table_cell),
            Paragraph("A fast algorithm that searches for exact words and keywords, just like pressing Ctrl+F on your keyboard.", table_cell),
        ],
        [
            Paragraph("<b>HNSW</b>", table_cell_bold),
            Paragraph("Hierarchical Navigable Small World", table_cell),
            Paragraph("A fast map inside the database that finds similar vector numbers in 2 milliseconds without checking every single row.", table_cell),
        ],
        [
            Paragraph("<b>RRF</b>", table_cell_bold),
            Paragraph("Reciprocal Rank Fusion", table_cell),
            Paragraph("A fair scoring formula that combines keyword search results and meaning search results without getting confused by different number scales.", table_cell),
        ],
        [
            Paragraph("<b>HyDE</b>", table_cell_bold),
            Paragraph("Hypothetical Document Embeddings", table_cell),
            Paragraph("The AI writes a fake sample answer first, and uses that sample answer to search real documents. Bridges the vocabulary gap.", table_cell),
        ],
        [
            Paragraph("<b>CRAG</b>", table_cell_bold),
            Paragraph("Corrective Retrieval-Augmented Generation", table_cell),
            Paragraph("The AI grades its own research: if the retrieved notes are good, it answers. If they are bad, it rewrites the question or cleans up messy sentences.", table_cell),
        ],
        [
            Paragraph("<b>RBAC</b>", table_cell_bold),
            Paragraph("Role-Based Access Control", table_cell),
            Paragraph("A digital security guard that checks user clearance (e.g. Intern vs Admin) so users never see confidential documents they aren't allowed to see.", table_cell),
        ],
        [
            Paragraph("<b>SSE</b>", table_cell_bold),
            Paragraph("Server-Sent Events", table_cell),
            Paragraph("A web connection that streams words one-by-one to your screen immediately, so you don't stare at a blank screen for 10 seconds.", table_cell),
        ],
        [
            Paragraph("<b>CI / CD</b>", table_cell_bold),
            Paragraph("Continuous Integration / Continuous Deployment", table_cell),
            Paragraph("An automated robot tester on GitHub that runs 82 tests on every pull request and blocks bad code from merging.", table_cell),
        ],
        [
            Paragraph("<b>OTel</b>", table_cell_bold),
            Paragraph("OpenTelemetry", table_cell),
            Paragraph("A digital stopwatch that measures how many milliseconds each step took (searching, database, generation) to find slow bottlenecks.", table_cell),
        ],
        [
            Paragraph("<b>PII</b>", table_cell_bold),
            Paragraph("Personally Identifiable Information", table_cell),
            Paragraph("Private user data like passwords, credit cards, emails, and phone numbers that our guardrails mask before processing.", table_cell),
        ],
        [
            Paragraph("<b>TTFT</b>", table_cell_bold),
            Paragraph("Time To First Token", table_cell),
            Paragraph("How fast the very first word appears on the user's screen (in ApexRAG: under 40 milliseconds).", table_cell),
        ],
    ]
    t_glossary = Table(glossary_data, colWidths=[0.8 * inch, 2.0 * inch, 4.2 * inch])
    t_glossary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), teal_light),
        ('BOX', (0, 0), (-1, -1), 1, border_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_glossary)
    story.append(Spacer(1, 14))

    # ========================== SECTION 3: WORD-FOR-WORD PITCH ==========================
    story.append(PageBreak())
    story.append(Paragraph("3. Word-for-Word Interview Scripts (Just Read or Memorize These)", h1_style))
    story.append(Paragraph("When the interviewer asks these common questions, read these exact words slowly and calmly:", body_style))

    story.append(Paragraph("A. When Asked: 'Tell me about yourself in 60 seconds'", h2_style))
    speech_1 = [
        [Paragraph(
            "<i>\"Hello! I am an AI and Software Engineer specializing in building enterprise Retrieval-Augmented Generation systems and production backend services. "
            "Recently, I architected <b>ApexRAG</b>, an enterprise-grade hybrid RAG platform benchmarked across 2,580 technical documentation sections. "
            "Instead of building a simple demo with off-the-shelf wrappers, I implemented syntax-aware AST chunking, multi-tenant RBAC security, "
            "Corrective RAG self-grading, and real-time streaming with OpenTelemetry observability. "
            "I enjoy solving tough challenges around hallucination reduction, latency optimization, and automated quality gates.\"</i>",
            script_style
        )]
    ]
    t_s1 = Table(speech_1, colWidths=[7.0 * inch])
    t_s1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_light),
        ('BOX', (0, 0), (-1, -1), 1, teal),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(t_s1)
    story.append(Spacer(1, 10))

    story.append(Paragraph("B. When Asked: 'Explain your ApexRAG project in 2 minutes'", h2_style))
    speech_2 = [
        [Paragraph(
            "<i>\"Most RAG projects on YouTube or resumes are basic tutorial scripts: fixed character chunking, a single vector search, and a direct OpenAI API call. "
            "They fail in real enterprise companies because of code-block corruption, security leaks, and hallucinations.<br/><br/>"
            "I built <b>ApexRAG</b> to solve these real-world production challenges using four key architectural pillars:<br/>"
            "<b>1. Syntax-Aware Parent-Document Chunking:</b> We use an AST state machine that keeps JSX and code blocks completely intact. We search small 100-word child chunks for high search precision, but return the full 600-word parent section to the LLM.<br/>"
            "<b>2. Multi-Tenant RBAC Security:</b> We enforce document-level clearance and tenant isolation before retrieval, so unauthorized users never see confidential company docs.<br/>"
            "<b>3. Corrective RAG (CRAG):</b> Our system self-grades retrieval relevance. If context is ambiguous, it performs sentence-level knowledge striping; if irrelevant, it triggers agentic query reformulation.<br/>"
            "<b>4. Production Serving & Observability:</b> We built an async FastAPI service with Server-Sent Events token streaming, PostgreSQL pgvector HNSW indexing, OpenTelemetry distributed tracing, and an automated GitHub Actions CI gate with 82 passing tests.\"</i>",
            script_style
        )]
    ]
    t_s2 = Table(speech_2, colWidths=[7.0 * inch])
    t_s2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_light),
        ('BOX', (0, 0), (-1, -1), 1, teal),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(t_s2)
    story.append(Spacer(1, 14))

    # ========================== SECTION 4: 10 CROSS QUESTIONS ==========================
    story.append(PageBreak())
    story.append(Paragraph("4. Top 10 Technical Cross-Questions (What Interviewers Will Ask You)", h1_style))
    story.append(Paragraph("Here is what the interviewer will challenge you on, and the exact senior-engineer answer to give:", body_style))

    qa_list = [
        ("Q1: Why did you build custom retrieval instead of using LangChain or LlamaIndex?",
         "LangChain and LlamaIndex introduce heavy dependency bloat and hidden synchronous blocking calls that freeze FastAPI's async event loop. We wrote lightweight, native components using native asyncio.gather, native Reciprocal Rank Fusion, and OpenTelemetry spans. This gives us sub-30ms retrieval latency, full transparency over memory allocation, and eliminated third-party framework lock-in."),

        ("Q2: Why did you choose React documentation as your benchmark corpus?",
         "React documentation has the single hardest problem in technical retrieval: temporal semantic drift. Legacy React uses class component lifecycles (componentDidMount), while Modern React uses hooks (useEffect). They describe the same concepts using completely different words. Standard vector search conflates them, while BM25 misses the synonymy. This made it the perfect testbed to prove why hybrid search with era filtering beats basic RAG."),

        ("Q3: What is AST chunking and why does naive character chunking fail?",
         "Naive chunking uses fixed character counts like 500 characters. In technical documentation, this cuts code blocks, functions, and JSX tags right in the middle, breaking the syntax and blinding the vector search. Our AST chunker tracks code fences (```) and Markdown header depth so code blocks remain 100% intact with language tags and parent-document IDs."),

        ("Q4: Why Reciprocal Rank Fusion (RRF) instead of just averaging BM25 and Vector scores?",
         "BM25 scores are unbounded numbers from 0 to infinity, while cosine similarity is bounded between -1 and 1. If you just add them or normalize them, you get score calibration skew across different query lengths. RRF is scale-agnostic and rank-based: RRF(d) = sum(1 / (60 + rank(d))). It reliably merges sparse keyword search and dense semantic search without manual weight tuning."),

        ("Q5: Why PostgreSQL with pgvector instead of Pinecone or Qdrant?",
         "Using an external vector database like Pinecone creates the Dual-Write Consistency Problem: relational user data lives in Postgres while vectors live in Pinecone, requiring complex distributed synchronization. PostgreSQL with pgvector lets us store metadata, user permissions, and HNSW vector embeddings in the exact same ACID-compliant database, executing filtered vector queries with connection pooling at zero SaaS cost."),

        ("Q6: What is Corrective RAG (CRAG) and how does it prevent hallucinations?",
         "Standard RAG naively assumes retrieved documents are always relevant. CRAG evaluates retrieved chunks into three grades: CORRECT, AMBIGUOUS, and INCORRECT. If documents are ambiguous, it performs Knowledge Striping to remove off-topic sentences. If documents are completely irrelevant, it triggers Agentic Query Reformulation rather than feeding false information to the LLM."),

        ("Q7: What is Role-Based Access Control (RBAC) in RAG and why is it critical?",
         "Without RBAC, any user can query the RAG and retrieve confidential files like executive salaries or internal security credentials. Our RBAC module tags chunks with tenant IDs and clearance levels (Public, Internal, Confidential, Restricted). Before retrieval happens, it filters out forbidden chunks at the database query level so unauthorized users never receive private data."),

        ("Q8: How many users can use your system at the same time?",
         "On a single 8-core server, our semantic cache serves over 1,500 requests per second for common queries in under 1 millisecond. For fresh, uncached RAG generation, our bounded concurrency semaphore (16 slots) supports 600 to 1,200 active concurrent users. Because our FastAPI service is completely stateless, we can scale horizontally across Kubernetes pods to handle tens of thousands of users."),

        ("Q9: What is your Automated CI/CD Quality Gate?",
         "In traditional software, tests only check if code compiles. In enterprise RAG, a code change can silently drop retrieval accuracy. Our GitHub Actions workflow (eval_gate.yml) runs ci_gate.py on every pull request, benchmarking against 100 verified Q&A pairs using Ragas Faithfulness and Recall. If scores drop below our SLA threshold, the pull request merge is automatically blocked."),

        ("Q10: What is Thompson Sampling in your Strategy Router?",
         "Instead of using a static if-else rule to pick between BM25, Semantic, Hybrid, and Rerank, our router uses a Bayesian Multi-Armed Bandit with Beta distributions. It balances exploitation (using the best known strategy) with exploration (testing under-utilized strategies). When users submit feedback through our POST /feedback API, the Beta posteriors update online in real-time without retraining from scratch."),
    ]

    for q, a in qa_list:
        q_box = [
            [Paragraph(f"<b>{q}</b>", table_cell_bold)],
            [Paragraph(f"<b>Answer:</b> {a}", table_cell)],
        ]
        t_q = Table(q_box, colWidths=[7.0 * inch])
        t_q.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg_light),
            ('BOX', (0, 0), (-1, -1), 0.5, border_color),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(t_q)
        story.append(Spacer(1, 5))

    # ========================== SECTION 5: NUMBERS TO MEMORIZE ==========================
    story.append(PageBreak())
    story.append(Paragraph("5. The Numbers & Metrics Cheat Sheet (Keep This Next to Your Screen)", h1_style))
    story.append(Paragraph("Keep these exact numbers memorized or written down on a sticky note during your interview:", body_style))

    metrics_data = [
        [Paragraph("<b>Metric / Stat</b>", table_cell_bold), Paragraph("<b>ApexRAG Value</b>", table_cell_bold), Paragraph("<b>Why It Matters</b>", table_cell_bold)],
        [
            Paragraph("Total Test Suite", table_cell_bold),
            Paragraph("<b>82 passed</b> (100% pass rate)", table_cell),
            Paragraph("Proves the entire codebase is verified across 18 test suites with zero broken paths.", table_cell),
        ],
        [
            Paragraph("Corpus Size", table_cell_bold),
            Paragraph("<b>2,580 documentation sections</b>", table_cell),
            Paragraph("Ingested from verified React Legacy and Modern git repositories with zero synthetic fake data.", table_cell),
        ],
        [
            Paragraph("Recall@4 Improvement", table_cell_bold),
            Paragraph("<b>+48.6% higher</b> than single vector", table_cell),
            Paragraph("Proves that Hybrid BM25 + pgvector HNSW + Cross-Encoder massively beats naive vector search.", table_cell),
        ],
        [
            Paragraph("Semantic Cache Latency", table_cell_bold),
            Paragraph("<b>< 1.0 millisecond</b> (Cosine >= 0.96)", table_cell),
            Paragraph("Reduces LLM inference costs and server CPU usage by 40% to 60% on recurring documentation queries.", table_cell),
        ],
        [
            Paragraph("Time-To-First-Token (TTFT)", table_cell_bold),
            Paragraph("<b>< 40 milliseconds</b>", table_cell),
            Paragraph("Delivers instant Server-Sent Events (SSE) streaming so users never wait for a frozen screen.", table_cell),
        ],
        [
            Paragraph("Infrastructure Cost", table_cell_bold),
            Paragraph("<b>$0.00 / Local CPU</b>", table_cell),
            Paragraph("Runs entirely on local hardware with zero external paid API subscriptions required for testing.", table_cell),
        ],
    ]
    t_metrics = Table(metrics_data, colWidths=[1.8 * inch, 2.0 * inch, 3.2 * inch])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), teal_light),
        ('BOX', (0, 0), (-1, -1), 1, border_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 14))

    # ========================== SECTION 6: LIVE DEMO STEPS ==========================
    story.append(Paragraph("6. Step-by-Step Live Demo Checklist (Screen Share)", h1_style))
    story.append(Paragraph("If the interviewer says: <i>'Can you share your screen and show me the project running?'</i> follow this exact sequence:", body_style))

    demo_steps = [
        ("Step 1: Open Terminal & Run Tests", "Run: <code>pytest tests/ -v</code>. Show that all <b>82 tests pass</b>. This immediately establishes credibility and shows you write professional automated test suites."),
        ("Step 2: Start Server & Open Studio", "Run: <code>uvicorn src.api.main:app --port 8000</code>. Open your browser at <code>http://localhost:8000/dashboard</code>. The interviewer will see your custom ApexRAG Studio."),
        ("Step 3: Ask a Technical Question", "Type: <i>'How to manage state in React?'</i>. Point to how the text streams word-by-word instantly using Server-Sent Events (SSE)."),
        ("Step 4: Show the OpenTelemetry Waterfall", "Click on the <b>Observability / Tracing</b> tab. Show the waterfall breakdown: explain how many milliseconds were spent on dense vector search, BM25, and LLM generation. Point to the itemized USD cost attribution."),
        ("Step 5: Show Multi-Tenant RBAC & CRAG", "Show how the response telemetry includes <code>rbac_telemetry</code> (confirming clearance levels) and <code>crag_telemetry</code> (confirming retrieval self-grading and knowledge refinement)."),
    ]
    for step_title, step_desc in demo_steps:
        s_box = [
            [Paragraph(f"<b>{step_title}</b>", table_cell_bold)],
            [Paragraph(step_desc, table_cell)],
        ]
        t_s = Table(s_box, colWidths=[7.0 * inch])
        t_s.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg_light),
            ('BOX', (0, 0), (-1, -1), 0.5, border_color),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(t_s)
        story.append(Spacer(1, 4))

    doc.build(story)
    print(f"PDF generated successfully at: {output_path}")


if __name__ == "__main__":
    generate_pitch_pdf()
