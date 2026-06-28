from litellm import acompletion 
from src.utils.config import get_model_config
from src.utils.obs import LLMUsageTracker
from src.utils.reasoning_extractor import extract_and_log_reasoning

class LitellmClient:
    def __init__(self):
        self.model = None
        self.team_config = None

    async def get_dynamic_llm_instance(self, team_id: str):
        print("This function called")
        try:
            async with get_model_config() as config:
                # Get the team's model configuration
                print("Team id is: ", team_id)
                team_config = await config.get_team_model_config(team_id)
                model = team_config["selected_model"]
                provider = team_config["provider"]
                provider_model = f"{provider}/{model}"
                model_config = team_config["config"]

                # Create LLM instance with the team's configuration
                llm_params = {
                    "model": provider_model,
                    **model_config
                }
                self.team_config = llm_params
                self.model = model
                return llm_params

        except Exception as e:
            print(f"Failed to create LLM instance for team {team_id}: {str(e)}")
            raise ValueError(f"Failed to get model configuration for team {team_id}: {str(e)}")
        
    
    async def generate_response(self, llm_params: dict, messages: list, auth_token: str) -> str:
        """
        Global async completion function that can be imported anywhere.
        Uses the dynamic LLM params from get_dynamic_llm_instance.

        Args:
            llm_params: Dict with "model" and provider config (from get_dynamic_llm_instance)
            messages: List of message dicts (e.g. [{"role": "user", "content": "..."}])
            **kwargs: Additional params passed to litellm.acompletion

        Returns:
            The content string from the LLM response.
        """
        token_tracker = LLMUsageTracker(auth_token=auth_token)
        response = await acompletion(
            messages=messages,
            **llm_params
        )
        token_tracker.track_response(response, model_name=llm_params.get('model', ""))
        result = response.choices[0].message.content
        result, _ = extract_and_log_reasoning(response_text=result, auth_token=auth_token)
        return result
    
