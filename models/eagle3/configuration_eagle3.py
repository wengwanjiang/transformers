from transformers import PretrainedConfig
from transformers import LlamaConfig
from ..auto.configuration_auto import AutoConfig
from typing import Any, Optional, Union
import os , json
class Eagle3Config(PretrainedConfig):

    model_type = "eagle3"

    def __init__(self, draft_config: str | LlamaConfig = None, target_config: str | LlamaConfig = None, length: int = 7, **kwargs):
        super().__init__(**kwargs)
        
        

        if isinstance(draft_config, str):
            self.draft_config = AutoConfig.from_pretrained(draft_config)
        elif isinstance(draft_config, LlamaConfig):
            self.draft_config = draft_config
        else:
            self.draft_config = LlamaConfig()
        if isinstance(target_config, str):
            self.target_config = target_config
        elif isinstance(target_config, LlamaConfig):
            self.target_config = target_config
        else:
            self.target_config = LlamaConfig()
        
        self.length = length
    

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        self.draft_config.length = self.length
        self.draft_config.save_pretrained(save_directory, push_to_hub, **kwargs)
        