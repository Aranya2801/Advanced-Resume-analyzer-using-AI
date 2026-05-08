"""
Generate sample resume dataset for testing.
Run: python generate_dataset.py
Outputs to data/sample_resumes/
"""

from pathlib import Path

SAMPLE_RESUMES = {
    "software_engineer_senior.txt": """ALEX CHEN
alex.chen@email.com | +1 (415) 555-0192 | linkedin.com/in/alexchen | github.com/alexchen | San Francisco, CA

PROFESSIONAL SUMMARY
Senior Software Engineer with 8+ years building large-scale distributed systems at FAANG-level companies. 
Led teams of 6-12 engineers, reduced infrastructure costs by $2.3M annually, and shipped products used by 50M+ users.
Passionate about ML systems, platform engineering, and developer tooling.

EXPERIENCE

Staff Software Engineer | Stripe, Inc. | San Francisco, CA | Jan 2021 – Present
• Architected a real-time fraud detection system processing 2.4M transactions/sec with <10ms P99 latency
• Led migration of 150 microservices from Kubernetes 1.18 → 1.28, reducing cluster costs by 34%
• Built internal ML feature platform serving 40+ data science teams, cutting model deployment time from 3 weeks to 2 days
• Mentored 6 junior engineers; 4 promoted within 18 months under my guidance
• Drove adoption of SLO-based alerting, reducing false-positive pages by 78%

Senior Software Engineer | Airbnb | San Francisco, CA | Mar 2018 – Jan 2021
• Redesigned search ranking pipeline using transformer-based embeddings, improving booking conversion by 11%
• Built A/B testing framework used by 200+ engineers for 500+ experiments annually
• Reduced API response time by 60% via intelligent caching and database query optimisation
• Authored RFC that standardised GraphQL schema design across 15 backend teams

Software Engineer | LinkedIn | Sunnyvale, CA | Jul 2016 – Mar 2018
• Developed recruiter matching algorithm serving 700M+ members with 99.99% uptime
• Optimised Kafka consumer groups reducing message lag from 45min to <2min during traffic spikes
• Shipped 3 features shipped to production with 0 P0 incidents

EDUCATION
B.S. Computer Science, Stanford University, 2016 | GPA: 3.9/4.0 | Honors: Phi Beta Kappa

SKILLS
Languages: Python, Go, Java, TypeScript, Scala, Rust
Distributed Systems: Kafka, Spark, Flink, Kubernetes, Istio, Envoy
Databases: PostgreSQL, MySQL, Redis, Cassandra, DynamoDB, Elasticsearch
ML/AI: PyTorch, TensorFlow, Hugging Face, MLflow, Vertex AI, SageMaker
Cloud: AWS (certified), GCP, Docker, Terraform, Helm
Observability: Prometheus, Grafana, Datadog, OpenTelemetry

CERTIFICATIONS
• AWS Solutions Architect – Professional (2023)
• Google Cloud Professional Data Engineer (2022)
• CKA: Certified Kubernetes Administrator (2021)

PROJECTS
• ResumeAI: Open-source resume analyzer with 2.1k GitHub stars
• DistSQL: Query planner optimisation library, 800+ npm downloads/week
""",

    "data_scientist_mid.txt": """PRIYA SHARMA
priya.sharma@ml.dev | +1 (628) 555-0847 | linkedin.com/in/priyasharma-ds | github.com/priyaml
New York, NY

SUMMARY
Data Scientist with 4 years specialising in NLP and recommendation systems.
Published 2 papers at NeurIPS. Reduced customer churn 23% at scale. 
Seeking a role combining research and production ML.

WORK EXPERIENCE

Data Scientist II | Spotify | New York, NY | Aug 2022 – Present
• Built podcast recommendation model increasing listen-through rate by 18% (impacting 100M+ users)
• Developed real-time content embedding pipeline using DistilBERT, reducing inference cost by 55%
• Created automated A/B test analysis framework saving 8 analyst-hours per experiment
• Collaborated with 3 cross-functional teams to define ML product roadmap

Data Scientist | American Express | New York, NY | Jun 2020 – Aug 2022
• Designed credit risk model (XGBoost ensemble) with AUC-ROC of 0.94, deployed in 6 countries
• Reduced false positive fraud alerts by 31% via SHAP-based feature refinement
• Mentored 2 junior data scientists; both received "Exceeds Expectations" ratings

Research Intern | DeepMind | London, UK | Jun 2019 – Aug 2019
• Contributed to multi-agent reinforcement learning research (1 publication)
• Implemented custom reward shaping for cooperative agents in StarCraft II environment

EDUCATION
M.S. Data Science, Columbia University, 2020 | GPA: 4.0
B.Tech Computer Science, IIT Bombay, 2018 | GPA: 9.1/10

PUBLICATIONS
• Sharma P. et al., "Efficient Cross-lingual Podcast Embeddings at Scale," NeurIPS 2023
• Sharma P. et al., "Robust Reward Shaping in Cooperative MARL," NeurIPS 2020 Workshop

TECHNICAL SKILLS
ML: PyTorch, TensorFlow, Hugging Face, XGBoost, LightGBM, scikit-learn, MLflow
NLP: BERT, GPT, LLaMA, LangChain, spaCy, NLTK
Data: Spark, dbt, Airflow, BigQuery, Snowflake, Pandas, Polars
Viz: Plotly, Matplotlib, Tableau, Looker
Cloud: GCP (Professional ML Engineer certified), AWS

AWARDS
• Spotify Hack Week Winner 2023 – "EchoMatch" personalised playlist system
• AmEx Innovation Award 2021 – Best ML Product
""",

    "frontend_engineer_entry.txt": """JAMIE RODRIGUEZ
jamie.rod@gmail.com | 617-555-9034 | github.com/jamierod | Boston, MA

OBJECTIVE
Recent CS graduate looking for a frontend engineering role. 
Strong in React and UI/UX. Built 8 full-stack projects.

EDUCATION
B.S. Computer Science, Boston University, May 2024
GPA: 3.7 / 4.0 | Dean's List (6 semesters)
Relevant Coursework: Data Structures, Algorithms, Web Engineering, HCI, Databases

PROJECTS

ResumeBuilder Pro | github.com/jamierod/resumebuilder
• Built drag-and-drop resume builder with React, TypeScript, and Tailwind CSS
• 400+ active users, 4.6/5 rating on Product Hunt
• Implemented PDF export using react-pdf, reducing file size by 40% vs competitors

StudyBuddy – AI Flashcard App | github.com/jamierod/studybuddy  
• Integrated OpenAI API for auto-generating flashcards from uploaded notes
• 1,200+ downloads on Google Play; 4.4-star rating
• Reduced user study time by 35% based on in-app surveys (n=150)

Portfolio Website | jamierod.dev
• Custom-built with Next.js 14, Three.js animations, Framer Motion
• 98/100 Lighthouse score; < 1.2s LCP

EXPERIENCE

Frontend Intern | Fidelity Investments | Boston, MA | Jun – Aug 2023
• Built 4 reusable React component library items used across 3 products
• Reduced page load time by 28% by implementing code splitting and lazy loading
• Participated in daily standups and sprint planning with 7-person agile team

Teaching Assistant – Web Engineering | Boston University | Jan – May 2024
• Guided 60 students through React, Node.js, and REST API assignments
• Held 4 office hours per week; average student satisfaction 4.8/5

SKILLS
Languages: JavaScript (ES2024), TypeScript, Python, HTML5, CSS3
Frameworks: React, Next.js, Vue 3, Svelte, Node.js, Express
Styling: Tailwind CSS, styled-components, Framer Motion
Tools: Git, Docker, Figma, Webpack, Vite, Jest, Playwright
Cloud: Vercel, Netlify, Firebase, basic AWS
""",
}

JOB_DESCRIPTIONS = {
    "senior_ml_engineer_jd.txt": """Senior Machine Learning Engineer – AI Platform Team

We are looking for a Senior ML Engineer to join our AI Platform team at TechCorp.

Responsibilities:
- Design and build scalable ML training and inference infrastructure
- Partner with research teams to productionise cutting-edge models
- Optimise model serving for latency and throughput at scale
- Lead technical design reviews and mentor junior engineers
- Contribute to ML platform roadmap

Requirements:
- 5+ years of software engineering experience, with 3+ years in ML engineering
- Expert-level Python; proficiency in Go or Rust a plus
- Deep experience with PyTorch and/or TensorFlow
- Experience with MLOps tools: MLflow, Kubeflow, or similar
- Strong understanding of distributed systems and Kubernetes
- Experience with LLMs, transformer fine-tuning, or RLHF a strong plus
- Track record of shipping ML systems to production at scale
- M.S. or Ph.D. in CS, Statistics, or related field preferred

Nice to Have:
- Experience with vLLM, TensorRT, or ONNX optimisation
- Publications at top ML venues (NeurIPS, ICML, ICLR)
- Open source contributions

We offer: $250K–$320K total comp, equity, remote-friendly, unlimited PTO.
""",

    "frontend_engineer_jd.txt": """Frontend Engineer – Consumer Products

Join our consumer product team to build delightful web experiences for millions of users.

Responsibilities:
- Build responsive, accessible React applications
- Collaborate with design team using Figma specs
- Write comprehensive unit and integration tests
- Optimise Core Web Vitals and performance metrics
- Participate in code reviews and contribute to our component library

Requirements:
- 2+ years of frontend engineering experience
- Expert knowledge of React and TypeScript
- Strong CSS skills including modern layout techniques
- Experience with testing: Jest, React Testing Library, Playwright
- Understanding of web performance optimisation
- Ability to translate designs into pixel-perfect implementations

Nice to Have:
- Next.js / SSR experience
- Animation libraries (Framer Motion, GSAP)
- Accessibility (WCAG 2.1) expertise
- Basic backend (Node.js/Python) knowledge

Compensation: $130K–$165K + equity + benefits
""",
}


def generate():
    resume_dir = Path("data/sample_resumes")
    jd_dir = Path("data/job_descriptions")
    resume_dir.mkdir(parents=True, exist_ok=True)
    jd_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in SAMPLE_RESUMES.items():
        path = resume_dir / filename
        path.write_text(content.strip())
        print(f"✅ Created {path}")

    for filename, content in JOB_DESCRIPTIONS.items():
        path = jd_dir / filename
        path.write_text(content.strip())
        print(f"✅ Created {path}")

    print(f"\n🎉 Dataset generated successfully!")
    print(f"   Resumes : {resume_dir}")
    print(f"   JDs     : {jd_dir}")


if __name__ == "__main__":
    generate()
