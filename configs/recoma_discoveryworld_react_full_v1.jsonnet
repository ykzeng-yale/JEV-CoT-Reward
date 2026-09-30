local model = std.extVar('MODEL');
local model_path = std.extVar('MODEL_PATH');
local output_dir = std.extVar('OUTPUT_DIR');
local max_env_calls = std.parseInt(std.extVar('MAX_ENV_CALLS'));
local seeds = std.parseJson(std.extVar('SEEDS'));
local scenario_prefixes = std.parseJson(std.extVar('SCENARIO_PREFIXES'));
{
    "renderers": [],
    "models": {
        "discoveryworld_init": {
            "type": "discoveryworld_loader",
            "threadid_offset": 0,
            "next_model": "react"
        },
        "react": {
            "type": "discoveryworld_react_controller",
            "action_model": "action",
            "observation_model": "environment",
            "add_roles": true,
            "max_output_length": 10000,
            "max_history": -1
        },
        "action": {
            "type": "discoveryworld_promptedlm",
            "prompt_file": "agents/recoma/prompts/react_prompt.txt",
            "generator_params": {
                "type": "hf_torch",
                "model": model,
                "model_path": model_path,
                "max_tokens": 400,
                "temperature": 0.0,
                "seed": 0,
                "stop": ["```\n"]
            }
        },
        "environment": {
            "type": "discoveryworld_env",
            "output_dir": null
        }
    },
    "search": {
        "type": "best_first",
        "start_model": "discoveryworld_init",
        "answerer": {
            "type": "discoveryworld_answerer",
            "output_dir": null
        },
        "stopping_conditions": [
            {"type": "max_env_calls", "max_env_calls": max_env_calls},
            {"type": "max_llm_calls", "max_llm_calls": 2 * max_env_calls + 2},
        ]
    },
    "reader": {
       "type": "discoveryworld_reader",
       "limit_prefixes": scenario_prefixes,
       "limit_seeds": seeds
    }
}
