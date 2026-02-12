# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multi-agent context rollout for offline conversation generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .router import BaseRouter


@dataclass
class Turn:
    """Represents a single turn in a multi-agent conversation."""

    agent_id: int
    prompt: str
    output: str
    context: str


class MultiAgentContextRollout:
    """Generates multi-agent conversations by routing turns to different agents."""

    def __init__(
        self,
        model_paths: Sequence[str],
        router: BaseRouter,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 256,
    ):
        """Initialize multi-agent context rollout.

        Args:
            model_paths: List of HuggingFace model paths (one per agent)
            router: Router instance to select agents
            device: Device for model inference ("cuda" or "cpu")
            dtype: Torch dtype for generation ("float16", "bfloat16", or "float32")
            max_new_tokens: Maximum tokens to generate per turn
        """
        self.model_paths = model_paths
        self.router = router
        self.device = device
        self.max_new_tokens = max_new_tokens

        # Parse dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(dtype.lower(), torch.bfloat16)

        # Lazy load models on first use
        self.models = {}
        self.tokenizers = {}
        print(f"[MultiAgentContextRollout] Initialized with {len(model_paths)} agents on {device}")

    def _load_agent(self, agent_id: int):
        """Lazy load model and tokenizer for an agent."""
        if agent_id in self.models:
            return

        model_path = self.model_paths[agent_id]
        print(f"[MultiAgentContextRollout] Loading agent {agent_id} from {model_path}...")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        model.eval()

        self.models[agent_id] = model
        self.tokenizers[agent_id] = tokenizer
        print(f"[MultiAgentContextRollout] Agent {agent_id} loaded successfully")

    @torch.no_grad()
    def generate(self, agent_id: int, context: str) -> str:
        """Generate response from a specific agent given context.

        Args:
            agent_id: Agent ID to generate from
            context: Conversation context (prompt + history)

        Returns:
            Generated response string
        """
        self._load_agent(agent_id)

        model = self.models[agent_id]
        tokenizer = self.tokenizers[agent_id]

        # Tokenize input
        inputs = tokenizer(context, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode only the generated part (exclude prompt)
        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0, prompt_length:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response.strip()

    def run(self, init_prompt: str, max_turns: int = 4) -> list[Turn]:
        """Generate a multi-agent conversation.

        Args:
            init_prompt: Initial prompt to start the conversation
            max_turns: Maximum number of turns in the conversation

        Returns:
            List of Turn objects representing the conversation
        """
        self.router.reset()
        turns: list[Turn] = []
        context = init_prompt

        for turn_idx in range(max_turns):
            # Select next agent
            agent_id = self.router.select_agent(context=context)

            # Determine current prompt (first turn uses init_prompt, rest use prev response)
            if turn_idx == 0:
                prompt = init_prompt
            else:
                prompt = turns[-1].output  # Previous agent's response

            # Generate response
            response = self.generate(agent_id=agent_id, context=context)

            # Create turn record
            turn = Turn(agent_id=agent_id, prompt=prompt, output=response, context=context)
            turns.append(turn)

            # Update context for next turn
            context = f"{context}\n{response}"

        return turns

    def __del__(self):
        """Cleanup models on deletion."""
        for model in self.models.values():
            del model
        torch.cuda.empty_cache()
