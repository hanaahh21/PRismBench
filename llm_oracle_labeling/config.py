# Configuration constants

# Risk type labels used for downstream training (multi-label risk classes)
RISK_TYPE_LABELS = [
    'Bug Risk',
    'Security Risk',
    'Performance Risk',
    'Maintainability Risk'
]

# Unary decision labels used during LLM prompting (4 prompts per PR)
UNARY_LABELS = [
    'Bug Risk',
    'Security Risk',
    'Performance Risk',
    'Maintainability Risk'
]

# Canonical CSV column names for unary outputs
LABEL_COLUMN_MAP = {
    'Bug Risk': 'bug',
    'Security Risk': 'security',
    'Performance Risk': 'performance',
    'Maintainability Risk': 'maintainability'
}

# Pipeline settings
MAX_ITEMS = None
BATCH_SIZE = 25
TARGET_ACCEPTED_COUNT = 50
MAX_CONVERGENCE_LAYERS = 20
FLUSH_EVERY = 50
LLM_SLEEP = 0.5
CONSENSUS_THRESHOLD = 3  # strict all-3 agreement

# Timeout settings
LLM_TIMEOUT = 30
OLLAMA_TIMEOUT = 900

# Prompt size limits
MAX_PROMPT_CHARS = 80000
MAX_DIFF_CHARS = 10000
MAX_COMMENTS_PER_ISSUE = 0
MAX_COMMENT_CHARS = 10000

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/chat"

# Model configuration
MODELS_CONFIG = [
    {'name': 'Mistral7BInstruct', 'model_id': 'mistral:7b-instruct', 'query_fn': 'query_mistral'},
    {'name': 'Llama3.1_8B', 'model_id': 'llama3.1:8b', 'query_fn': 'query_llama'},
    {'name': 'Gemma2', 'model_id': 'gemma2:9b', 'query_fn': 'query_gemma'}
]

# Directory / file configuration
LAYERS_BASE_DIR = 'Layers'
LAYER_CSV_SUBDIR = 'csv_files'
PER_LAYER_MODEL_FILE_PATTERN = 'Layer{layer}_{model_name}.csv'
LAYER_ACCEPTED_FILE_PATTERN = 'Layer{layer}_accepted.csv'
LAYER_UNLABELED_FILE_PATTERN = 'Layer{layer}_unlabeled.csv'
GLOBAL_ACCEPTED_FILE = 'accepted.csv'
GLOBAL_UNLABELED_FILE = 'unlabeled.csv'

# Legacy output config kept for backward compatibility
ACCEPTED_LABELS_DIR = 'AcceptedLabels'
HUMAN_REVIEW_DIR = 'HumanEscalation'
MODEL_OUTPUTS_BASE_DIR = 'ModelIntermediateOutputs'
PER_MODEL_FILE_PATTERN = '{model_name}_predictions_{loop_number}.csv'
ACCEPTED_FILE_PATTERN = 'accepted_labels_{loop_number}.csv'
HUMAN_REVIEW_FILE_PATTERN = 'human_needed_{loop_number}.csv'
