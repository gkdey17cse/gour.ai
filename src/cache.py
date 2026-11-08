from sentence_transformers import SentenceTransformer, util

# Predefined intent dataset for gour.ai
INTENT_CACHE = [
    {
        "intent": "greetings",
        "sample_queries": [
            "Hi Gour",
            "Hello Gour Krishna",
            "Hey",
            "Good morning",
            "Hi there",
            "Who are you",
            "Introduce yourself",
        ],
        "response": (
            "Hello! I am gour.ai, an AI representative for Gour Krishna Dey. "
            "I can answer questions regarding Gour's experience in Agentic AI, RAG architectures, "
            "data engineering at EXL, research at IIIT Delhi, and technical projects. How can I help you today?"
        ),
    },
    {
        "intent": "current_role",
        "sample_queries": [
            "What is your current designation?",
            "Where do you work?",
            "What is your current job?",
            "What is Gour's role at EXL?",
            "Current company and role",
        ],
        "response": (
            "Gour Krishna Dey is currently working as a Developer (B1) at EXL Services "
            "in the Dataharbor Team, focusing on production-grade Agentic AI workflows, "
            "RAG architectures, and SAS-to-Python data lineage pipelines."
        ),
    },
    {
        "intent": "current_project",
        "sample_queries": [
            "What is your current project?",
            "What are you working on right now?",
            "Tell me about your project at EXL",
            "What project is Gour building currently?",
        ],
        "response": (
            "At EXL Services, Gour is currently working on enhancing a Python-based data lineage pipeline "
            "that traces source-to-target data flows across enterprise SAS assets, as well as developing "
            "Agentic AI workflows using Agent Development Kits (ADK) and RAG."
        ),
    },
    {
        "intent": "education",
        "sample_queries": [
            "What is your educational background?",
            "Where did you graduate from?",
            "Tell me about your M.Tech degree",
            "Which college did Gour attend?",
        ],
        "response": (
            "Gour completed his M.Tech in Computer Science and Engineering from IIIT Delhi (CGPA: 8.00/10.0) "
            "and holds a B.Tech in CSE from Govt. College of Engineering & Textile Technology, Serampore (DGPA: 9.38/10.0)."
        ),
    },
    {
        "intent": "contact",
        "sample_queries": [
            "How can I contact Gour?",
            "What is your email address?",
            "LinkedIn profile link",
            "How to reach out to Gour Krishna Dey?",
        ],
        "response": (
            "You can reach Gour Krishna Dey via email at gkdey17cse@gmail.com. "
            "You can also connect with him on LinkedIn and GitHub through his portfolio."
        ),
    },
]


class SemanticCache:
    def __init__(self, embedder: SentenceTransformer, threshold: float = 0.82):
        self.embedder = embedder
        self.threshold = threshold
        self.cached_vectors = []
        self._initialize_cache()

    def _initialize_cache(self):
        """Pre-computes embeddings for all sample intent queries on application start."""
        for item in INTENT_CACHE:
            embeddings = self.embedder.encode(
                item["sample_queries"], convert_to_tensor=True
            )
            self.cached_vectors.append(
                {
                    "intent": item["intent"],
                    "response": item["response"],
                    "embeddings": embeddings,
                }
            )

    def check_cache(self, user_query: str):
        """
        Computes cosine similarity of user query against all cached sample queries.
        Returns response dict if highest score >= threshold, otherwise returns None.
        """
        query_vector = self.embedder.encode(user_query, convert_to_tensor=True)
        best_score = 0.0
        best_response = None
        matched_intent = None

        for item in self.cached_vectors:
            # Cosine similarity matrix between query and intent sample queries
            cosine_scores = util.cos_sim(query_vector, item["embeddings"])[0]
            max_score_for_intent = float(cosine_scores.max())

            if max_score_for_intent > best_score:
                best_score = max_score_for_intent
                best_response = item["response"]
                matched_intent = item["intent"]

        if best_score >= self.threshold:
            return {
                "answer": best_response,
                "provider": f"Semantic Cache (Intent: {matched_intent})",
                "similarity_score": round(best_score, 4),
                "is_cached": True,
            }

        return None
