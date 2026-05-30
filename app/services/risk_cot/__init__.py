from .prompt_engine import PromptEngine
from .inference_engine import InferenceEngine
from .inspector_engine import InspectorEngine

# Global instances for stateful services
inference_engine = InferenceEngine()
inspector_engine = InspectorEngine()
