"""Role profiles used to generate the synthetic benchmark.

Each role has core skills (almost every resume/JD for it mentions them),
optional skills, and bullet templates. A template is (text, skills that make
sense in its {skill} slot); numeric slots like {pct} or {ms} get random but
plausible values.

PARAPHRASES are ways people describe a skill *without* naming it in a way
the taxonomy recognises. generate.py checks that the taxonomy really does
miss each one, so the "paraphrase" resumes test what they're meant to test.
"""

ROLES = {
    "ml_engineer": {
        "titles": ["Machine Learning Engineer", "ML Engineer", "Applied ML Engineer", "ML Platform Engineer"],
        "core": ["Python", "PyTorch", "Machine Learning", "MLflow", "Docker", "Kubernetes"],
        "optional": ["TensorFlow", "Model Serving", "ONNX", "AWS", "Spark", "Feature Engineering",
                     "Kubeflow", "Triton Inference Server", "Fine-tuning", "LLM", "Go", "Airflow"],
        "bullets": [
            ("Trained and deployed ranking models with {skill}, lifting click-through rate by {pct}%", ["PyTorch", "TensorFlow", "Machine Learning"]),
            ("Built a feature store and batch pipeline in {skill} serving {n}+ models in production", ["Spark", "Python", "Airflow"]),
            ("Cut p99 inference latency from {ms}ms to {ms2}ms by moving serving to {skill}", ["ONNX", "Triton Inference Server", "Kubernetes", "Go"]),
            ("Set up experiment tracking and model registry with {skill} for {n} data scientists", ["MLflow", "Kubeflow"]),
            ("Packaged training jobs with {skill}, reducing time-to-deploy from {d} days to {d2}", ["Docker", "Kubernetes", "Kubeflow"]),
            ("Wrote offline evaluation tooling in {skill} that caught {pct}% of model regressions before release", ["Python"]),
            ("Fine-tuned {skill} models on {n}k labelled examples for intent classification", ["PyTorch", "LLM", "TensorFlow"]),
            ("Owned the retraining pipeline for fraud models scoring {n}M events per day on {skill}", ["AWS", "Kubernetes", "Spark"]),
        ],
    },
    "data_scientist": {
        "titles": ["Data Scientist", "Senior Data Scientist", "Product Data Scientist"],
        "core": ["Python", "SQL", "Statistics", "scikit-learn", "Pandas", "A/B Testing"],
        "optional": ["XGBoost", "Machine Learning", "Tableau", "R", "Spark", "Time Series",
                     "PyTorch", "Matplotlib", "BigQuery", "SHAP", "Recommender Systems"],
        "bullets": [
            ("Designed and analysed {n} {skill} on checkout flows, shipping changes worth {pct}% conversion", ["A/B Testing"]),
            ("Built a churn model with {skill} (AUC {auc}) used to target retention offers", ["scikit-learn", "XGBoost", "PyTorch"]),
            ("Forecasted weekly demand for {n} stores with {skill}, reducing stock-outs by {pct}%", ["Time Series", "Statistics", "R"]),
            ("Defined north-star metrics with product managers and built dashboards in {skill}", ["Tableau", "Matplotlib"]),
            ("Explained model decisions to the risk team using {skill}, getting sign-off for launch", ["SHAP"]),
            ("Cleaned and joined {n}M rows of transaction data in {skill} for pricing analysis", ["Pandas", "SQL", "Spark", "BigQuery"]),
            ("Ran causal impact analysis on a marketing campaign using {skill}, estimating {pct}% incremental lift", ["Statistics", "Python", "R"]),
        ],
    },
    "data_engineer": {
        "titles": ["Data Engineer", "Senior Data Engineer", "Analytics Engineer"],
        "core": ["Python", "SQL", "Spark", "Airflow", "ETL", "Kafka"],
        "optional": ["dbt", "Snowflake", "BigQuery", "AWS", "Databricks", "Data Warehousing",
                     "Scala", "Terraform", "Flink", "PostgreSQL", "Docker"],
        "bullets": [
            ("Built batch and streaming pipelines in {skill} moving {n}TB per day into the warehouse", ["Spark", "Flink", "Python", "Scala"]),
            ("Migrated {n} legacy cron jobs to {skill}, cutting pipeline failures by {pct}%", ["Airflow"]),
            ("Modelled the core sales tables in {skill}, used by {n} analysts daily", ["dbt", "SQL", "Snowflake", "Data Warehousing"]),
            ("Reduced {skill} spend by ${k}k a year by partitioning and clustering large tables", ["Snowflake", "BigQuery", "Databricks"]),
            ("Implemented data quality checks in the {skill} layer that flag schema drift within {m} minutes", ["ETL"]),
            ("Ingested clickstream events from {skill} with exactly-once delivery guarantees", ["Kafka"]),
        ],
    },
    "backend_engineer": {
        "titles": ["Backend Engineer", "Software Engineer (Backend)", "Senior Backend Engineer"],
        "core": ["Java", "PostgreSQL", "REST", "Microservices", "Docker", "Redis"],
        "optional": ["Go", "Spring Boot", "Kafka", "AWS", "Kubernetes", "gRPC", "GraphQL",
                     "Python", "Distributed Systems", "Elasticsearch", "CI/CD"],
        "bullets": [
            ("Designed {skill} services handling {n}k requests per second at {ms}ms p99", ["Java", "Go", "Spring Boot", "gRPC"]),
            ("Split the monolith into {n} {skill}, cutting deploy time by {pct}%", ["Microservices"]),
            ("Added a caching layer with {skill} that removed {pct}% of database reads", ["Redis"]),
            ("Built the payments {skill} API used by {n} partner integrations", ["REST", "GraphQL"]),
            ("Led the move to {skill}, which brought on-call pages down by {pct}%", ["Kubernetes", "Docker", "AWS", "Kafka"]),
            ("Wrote load tests and fixed lock contention in {skill}, raising throughput {x}x", ["PostgreSQL", "Elasticsearch"]),
        ],
    },
    "frontend_engineer": {
        "titles": ["Frontend Engineer", "Front-End Developer", "UI Engineer"],
        "core": ["JavaScript", "TypeScript", "React", "CSS", "HTML", "Jest"],
        "optional": ["Next.js", "Redux", "Tailwind CSS", "Accessibility", "Web Performance",
                     "Playwright", "Figma", "GraphQL", "Webpack", "Vue", "Framer Motion"],
        "bullets": [
            ("Rebuilt the checkout UI in {skill}, improving conversion by {pct}%", ["React", "TypeScript", "Next.js", "Vue"]),
            ("Built a shared {skill} component library used across {n} product teams", ["React", "TypeScript"]),
            ("Improved Largest Contentful Paint from {s}s to {s2}s on the landing pages with {skill}", ["Web Performance", "Webpack", "Next.js"]),
            ("Raised test coverage of the web app from {pct}% to {pct2}% with {skill}", ["Jest", "Playwright"]),
            ("Fixed {n} {skill} issues to reach WCAG AA on the signup flow", ["Accessibility"]),
            ("Implemented design system tokens from {skill} specs in plain {skill2}", ["Figma"]),
            ("Rewrote legacy page layouts in {skill}, deleting {n}k lines of old styles", ["CSS", "Tailwind CSS", "HTML"]),
        ],
    },
    "fullstack_engineer": {
        "titles": ["Full Stack Engineer", "Full-Stack Developer", "Software Engineer (Full Stack)"],
        "core": ["JavaScript", "TypeScript", "React", "Node.js", "PostgreSQL", "REST"],
        "optional": ["Next.js", "Express", "MongoDB", "Docker", "AWS", "GraphQL", "Redis",
                     "Python", "Django", "CI/CD", "Tailwind CSS"],
        "bullets": [
            ("Shipped an end-to-end booking feature with {skill} used by {n}k customers", ["React", "Node.js", "Next.js", "TypeScript"]),
            ("Built internal admin tools in {skill} that saved support {n} hours a week", ["React", "Django", "Express"]),
            ("Designed the {skill} schema and API for the new subscriptions product", ["PostgreSQL", "MongoDB", "GraphQL"]),
            ("Cut page load time by {pct}% with server-side rendering and {skill} caching", ["Redis", "Next.js"]),
            ("Set up preview deployments for every pull request with {skill}, speeding up reviews", ["Docker", "CI/CD", "AWS"]),
        ],
    },
    "devops_engineer": {
        "titles": ["DevOps Engineer", "Site Reliability Engineer", "Platform Engineer"],
        "core": ["Kubernetes", "Terraform", "AWS", "Docker", "CI/CD", "Linux"],
        "optional": ["Prometheus", "Grafana", "Helm", "Ansible", "Python", "Go", "ArgoCD",
                     "GitHub Actions", "Datadog", "Bash", "GCP"],
        "bullets": [
            ("Managed {n} production {skill} clusters with {pct2}% uptime", ["Kubernetes"]),
            ("Wrote {skill} modules for all shared infrastructure, removing manual changes", ["Terraform", "Ansible", "Helm"]),
            ("Cut {skill} costs by ${k}k a year through rightsizing and spot instances", ["AWS", "GCP"]),
            ("Built {skill} pipelines that run in under {m} minutes", ["CI/CD", "GitHub Actions", "ArgoCD"]),
            ("Set up SLO alerting in {skill}, reducing noisy pages by {pct}%", ["Prometheus", "Grafana", "Datadog"]),
            ("Hardened {skill} hosts and wrote runbooks for the {n} most common failures", ["Linux", "Bash", "Docker"]),
        ],
    },
    "nlp_scientist": {
        "titles": ["NLP Research Scientist", "Research Scientist, NLP", "Applied Scientist (NLP)"],
        "core": ["Python", "PyTorch", "NLP", "Transformers", "Deep Learning", "Research"],
        "optional": ["Hugging Face", "LLM", "Fine-tuning", "RLHF", "BERT", "JAX", "RAG",
                     "spaCy", "Statistics", "Machine Learning", "LangChain"],
        "bullets": [
            ("Published {n} {skill} papers on low-resource machine translation at ACL and EMNLP", ["NLP", "Research"]),
            ("Pre-trained a {n}B parameter language model on {n2}B tokens with {skill}", ["PyTorch", "JAX", "Deep Learning"]),
            ("Improved {skill} retrieval accuracy by {pct}% with a hybrid dense-sparse retriever", ["RAG", "NLP"]),
            ("Fine-tuned {skill} models for summarisation, beating the baseline by {r} ROUGE", ["Transformers", "BERT", "LLM", "Hugging Face"]),
            ("Built evaluation suites in {skill} for factuality and toxicity across {n} languages", ["Python", "NLP"]),
            ("Mentored {n} PhD interns on {skill} and instruction-tuning projects", ["RLHF", "Fine-tuning", "Deep Learning"]),
        ],
    },
    "data_analyst": {
        "titles": ["Data Analyst", "Business Analyst", "BI Analyst"],
        "core": ["SQL", "Tableau", "Excel", "Statistics", "Power BI", "Python"],
        "optional": ["Pandas", "Looker", "BigQuery", "A/B Testing", "R", "Snowflake",
                     "Communication", "Stakeholder Management"],
        "bullets": [
            ("Built {n} {skill} dashboards tracking revenue and retention for leadership", ["Tableau", "Power BI", "Looker"]),
            ("Automated the weekly sales report with {skill}, saving {n} hours of manual work", ["Python", "Excel", "SQL"]),
            ("Analysed customer survey data in {skill} and presented findings to {n} stakeholders", ["Excel", "R", "Pandas"]),
            ("Wrote {skill} to reconcile billing data, finding ${k}k in missed invoices", ["SQL"]),
            ("Measured campaign ROI across {n} channels using {skill}", ["Statistics", "A/B Testing", "BigQuery"]),
        ],
    },
    "qa_engineer": {
        "titles": ["QA Engineer", "Test Automation Engineer", "SDET"],
        "core": ["Selenium", "Python", "Unit Testing", "CI/CD", "Playwright", "Jest"],
        "optional": ["Cypress", "Java", "pytest", "JavaScript", "REST", "Docker", "Agile",
                     "GitHub Actions", "SQL"],
        "bullets": [
            ("Automated {n} regression test cases with {skill}, cutting release testing from {d} days to {d2}", ["Selenium", "Playwright", "Cypress"]),
            ("Built an API test suite in {skill} that runs on every merge in {m} minutes", ["pytest", "Python", "Jest"]),
            ("Found and triaged {n} bugs before the v2 launch through {skill}", ["Unit Testing", "Agile"]),
            ("Introduced flaky-test tracking in {skill}, reducing reruns by {pct}%", ["CI/CD", "GitHub Actions"]),
        ],
    },
    # Non-technical roles: never relevant to any JD in the benchmark, but used
    # as the "true" background of keyword-stuffed resumes.
    "marketing_manager": {
        "titles": ["Marketing Manager", "Growth Marketing Manager", "Digital Marketing Lead"],
        "core": ["Communication", "Stakeholder Management", "Project Management"],
        "optional": ["Excel", "Leadership"],
        "bullets": [
            ("Planned and ran {n} paid social campaigns with a ${k}k quarterly budget", []),
            ("Grew newsletter subscribers by {pct}% through referral incentives", []),
            ("Managed relationships with {n} agencies and freelance designers", []),
            ("Launched the brand refresh across web, email and events", []),
            ("Increased organic traffic by {pct}% with a content calendar and SEO fixes", []),
        ],
    },
    "operations_manager": {
        "titles": ["Operations Manager", "Supply Chain Coordinator", "Office Manager"],
        "core": ["Project Management", "Leadership", "Communication"],
        "optional": ["Excel", "Stakeholder Management"],
        "bullets": [
            ("Coordinated logistics for {n} warehouses and {n2} delivery partners", []),
            ("Reduced order processing time by {pct}% by redesigning the intake process", []),
            ("Hired and trained {n} customer support staff", []),
            ("Negotiated vendor contracts saving ${k}k annually", []),
        ],
    },
}

TECH_ROLES = [r for r in ROLES if r not in ("marketing_manager", "operations_manager")]

# Graded relevance between a JD's role and a resume's true role.
# 2 = same role, 1 = adjacent (a recruiter would plausibly consider them).
ADJACENT = {
    ("ml_engineer", "data_scientist"), ("ml_engineer", "nlp_scientist"),
    ("ml_engineer", "data_engineer"), ("data_scientist", "data_analyst"),
    ("data_scientist", "nlp_scientist"), ("data_engineer", "backend_engineer"),
    ("backend_engineer", "fullstack_engineer"), ("frontend_engineer", "fullstack_engineer"),
    ("backend_engineer", "devops_engineer"), ("qa_engineer", "frontend_engineer"),
}


def relevance(jd_role: str, resume_role: str) -> int:
    if jd_role == resume_role:
        return 2
    if (jd_role, resume_role) in ADJACENT or (resume_role, jd_role) in ADJACENT:
        return 1
    return 0


# Descriptions of a skill that a recruiter would recognise but the taxonomy
# doesn't. generate.py drops any entry the taxonomy *does* catch.
PARAPHRASES = {
    "Kubernetes": ["EKS clusters", "container orchestration on GKE", "pod autoscaling and cluster upgrades"],
    "Docker": ["containerised services", "container images for every service"],
    "PyTorch": ["neural networks written with the Torch autograd API", "Lightning training loops"],
    "TensorFlow": ["Keras-style graph models on TPU"],
    "MLflow": ["an experiment tracking server and model registry"],
    "Machine Learning": ["predictive modelling", "supervised models"],
    "Python": ["Jupyter-based tooling and scripts in a dynamically typed language"],
    "SQL": ["complex joins and window functions over relational tables", "analytical queries"],
    "Statistics": ["hypothesis tests and confidence intervals", "regression analysis and significance testing"],
    "A/B Testing": ["controlled online experiments", "randomised experiments on product changes"],
    "scikit-learn": ["gradient-boosted and random forest models in sklearn-style pipelines"],
    "Pandas": ["dataframe wrangling"],
    "Spark": ["distributed dataframe jobs on EMR"],
    "Airflow": ["DAG-based job scheduling"],
    "ETL": ["ingestion and transformation jobs"],
    "Kafka": ["event streaming with topics and consumer groups"],
    "Java": ["JVM services"],
    "PostgreSQL": ["a relational database with heavy indexing and query tuning"],
    "REST": ["HTTP JSON APIs"],
    "Microservices": ["small independently deployed services"],
    "Redis": ["an in-memory key-value cache"],
    "React": ["hooks-based component UIs"],
    "TypeScript": ["statically typed JS"],
    "CSS": ["responsive layouts with flexbox and grid"],
    "Jest": ["snapshot and unit tests for components"],
    "Terraform": ["infrastructure as code with HCL"],
    "AWS": ["EC2, S3 and IAM"],
    "CI/CD": ["automated build and deploy pipelines"],
    "Linux": ["systemd services and kernel tuning"],
    "NLP": ["text classification and named entity tagging", "language understanding models"],
    "Transformers": ["attention-based sequence models"],
    "Deep Learning": ["neural network training at scale"],
    "Tableau": ["interactive BI dashboards"],
    "Power BI": ["DAX measures and BI reports"],
    "Selenium": ["browser automation with WebDriver"],
    "Unit Testing": ["test-driven development"],
    "Node.js": ["server-side JavaScript"],
}

FIRST_NAMES = ["Aarav", "Maya", "Liam", "Sofia", "Kenji", "Amara", "Lucas", "Isha", "Noah", "Elena",
               "Omar", "Chloe", "Ravi", "Hana", "Mateo", "Zara", "Ethan", "Priya", "Daniel", "Leila",
               "Arjun", "Grace", "Tomas", "Nia", "Felix", "Ananya", "Owen", "Mei", "Samuel", "Rhea"]
LAST_NAMES = ["Sharma", "Okafor", "Nguyen", "Garcia", "Tanaka", "Muller", "Patel", "Kowalski", "Silva",
              "Chen", "Haddad", "Novak", "Iyer", "Rossi", "Andersen", "Das", "Kim", "Mensah", "Ali", "Brown"]
COMPANIES = ["Northwind", "Brightpath", "Helix Labs", "Quantia", "Bluefin Systems", "Orbital Retail",
             "Kestrel Health", "Lumen Bank", "Fieldstone", "Paperkite", "Redwood Analytics", "Nimbus Logistics",
             "Saffron Media", "Tidewater Insurance", "Copperleaf", "Atlas Mobility"]
SCHOOLS = ["University of Michigan", "IIT Delhi", "University of Toronto", "TU Munich", "NUS",
           "Georgia Tech", "University of Melbourne", "KIIT University", "UCL", "Purdue University"]
DEGREES = {
    "tech": ["B.Tech Computer Science", "B.S. Computer Science", "M.S. Computer Science",
             "B.E. Information Technology", "M.S. Data Science", "B.S. Electrical Engineering"],
    "nontech": ["BBA Marketing", "B.A. Communications", "MBA", "B.Com"],
}
