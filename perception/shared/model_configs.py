"""
Unified model configurations for all perception tests.

All test runners can import and use MODEL_CONFIGS from this module.

IMPORTANT: Before using, replace "YOUR_API_KEY_HERE" with your actual API keys
for the respective services (Alibaba DashScope, Google, Zhipu AI).
"""

MODEL_CONFIGS = {
    # Qwen3-VL models (Alibaba DashScope)
    "qwen3-vl-8b": {
        "api_key": "YOUR_API_KEY_HERE",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen3-vl-8b-instruct",
    },
    "qwen3-vl-8b-thinking": {
        "api_key": "YOUR_API_KEY_HERE",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen3-vl-8b-thinking",
    },
    "qwen3-vl-30b": {
        "api_key": "YOUR_API_KEY_HERE",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen3-vl-30b-a3b-instruct",
    },
    "qwen3-vl-30b-a3b-instruct": {
        "api_key": "YOUR_API_KEY_HERE",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen3-vl-30b-a3b-instruct",
    },
    "qwen3-vl-235b": {
        "api_key": "YOUR_API_KEY_HERE",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen3-vl-235b-a22b-instruct",
    },
    "qwen3-vl-plus": {
        "api_key": "YOUR_API_KEY_HERE",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen3-vl-plus",
    },
    # Google models (Gemini/Gemma)
    "gemma3": {
        "api_key": "YOUR_API_KEY_HERE",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model_name": "gemma-3-27b-it",
    },
    "gemma-3-27b": {
        "api_key": "YOUR_API_KEY_HERE",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model_name": "gemma-3-27b-it",
    },
    "gemini-2.5-flash-lite": {
        "api_key": "YOUR_API_KEY_HERE",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model_name": "gemini-2.5-flash-lite",
    },
    "gemini-3-pro-preview": {
        "api_key": "YOUR_API_KEY_HERE",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model_name": "gemini-3-pro-preview",
    },
    # GLM models (Zhipu AI)
    "glm4v-thinking": {
        "api_key": "YOUR_API_KEY_HERE",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "model_name": "glm-4.1v-thinking-flash",
    },
    "glm-4.5v": {
        "api_key": "YOUR_API_KEY_HERE",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "model_name": "glm-4.5v",
    },
}
