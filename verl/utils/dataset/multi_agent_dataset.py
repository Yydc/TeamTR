# Multi-agent multi-turn dataset utilities for TeamTR.

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Optional

import datasets
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


def _truncate_ids(ids: list[int], max_len: int, truncation: str) -> list[int]:
    if len(ids) <= max_len:
        return ids
    if truncation == "left":
        return ids[-max_len:]
    if truncation == "right":
        return ids[:max_len]
    if truncation == "middle":
        left_half = max_len // 2
        right_half = max_len - left_half
        return ids[:left_half] + ids[-right_half:]
    if truncation == "error":
        raise RuntimeError(f"Sequence length {len(ids)} exceeds max_len={max_len}.")
    raise ValueError(f"Unsupported truncation mode: {truncation}")


class MultiAgentTurnDataset(Dataset):
    """Multi-turn dataset that yields context/response tensors plus agent_id."""

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/multi_agent"))
        self.context_key = config.get("context_key", "context")
        self.response_key = config.get("response_key", "response")
        self.prompt_key = config.get("prompt_key", "prompt")
        self.agent_id_key = config.get("agent_id_key", "agent_id")
        self.turn_id_key = config.get("turn_id_key", "turn_id")
        self.conversation_id_key = config.get("conversation_id_key", "conversation_id")

        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.max_response_length = config.get("max_response_length", 512)
        self.truncation = config.get("truncation", "error")
        self.context_left_pad = config.get("context_left_pad", True)
        self.response_left_pad = config.get("response_left_pad", False)

        self._download()
        self._read_files_and_prepare()

    def _download(self):
        for i, data_file in enumerate(self.data_files):
            self.data_files[i] = copy_to_local(src=data_file, cache_dir=self.cache_dir)

    def _read_files_and_prepare(self):
        dataframes = []
        for data_file in self.data_files:
            suffix = Path(data_file).suffix
            if suffix in [".json", ".jsonl"]:
                dataframe = datasets.load_dataset("json", data_files=data_file)["train"]
            elif suffix == ".parquet":
                dataframe = datasets.load_dataset("parquet", data_files=data_file)["train"]
            else:
                raise ValueError(f"Unsupported data file format: {data_file}")
            dataframes.append(dataframe)

        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
        self.records = None

        if self.context_key not in self.dataframe.column_names:
            self._build_context_records()

    def _build_context_records(self):
        required = {self.conversation_id_key, self.turn_id_key, self.prompt_key, self.response_key}
        if not required.issubset(set(self.dataframe.column_names)):
            raise ValueError(
                "Multi-turn format requires context_key or "
                "conversation_id/turn_id/prompt/response columns."
            )

        df = self.dataframe.to_pandas()
        records = []
        for conv_id, group in df.groupby(self.conversation_id_key):
            group = group.sort_values(self.turn_id_key)
            history: list[str] = []
            for _, row in group.iterrows():
                prompt = row.get(self.prompt_key, "") or ""
                response = row.get(self.response_key, "") or ""
                if prompt:
                    context = "\n".join(history + [prompt])
                else:
                    context = "\n".join(history)
                records.append(
                    {
                        self.context_key: context,
                        self.response_key: response,
                        self.agent_id_key: int(row.get(self.agent_id_key, -1)),
                        self.turn_id_key: int(row.get(self.turn_id_key, 0)),
                        self.conversation_id_key: row.get(self.conversation_id_key, conv_id),
                    }
                )
                history.extend([prompt, response])

        self.records = records

    def __len__(self):
        if self.records is not None:
            return len(self.records)
        return len(self.dataframe)

    def __getitem__(self, item):
        if self.records is not None:
            row_dict = self.records[item]
        else:
            row_dict = self.dataframe[item]

        context_text = row_dict.get(self.context_key, "") or ""
        response_text = row_dict.get(self.response_key, "") or ""
        agent_id = int(row_dict.get(self.agent_id_key, -1))

        context_ids = self.tokenizer(context_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        response_ids = self.tokenizer(response_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

        context_ids_list = _truncate_ids(context_ids.tolist(), self.max_prompt_length, self.truncation)
        response_ids_list = _truncate_ids(response_ids.tolist(), self.max_response_length, self.truncation)
        raw_prompt_ids = context_ids_list

        context_ids_raw = torch.tensor(context_ids_list, dtype=torch.long)
        response_ids_raw = torch.tensor(response_ids_list, dtype=torch.long)

        # padded context/response for batching
        ctx_input_ids = context_ids_raw.unsqueeze(0)
        ctx_attention_mask = torch.ones_like(ctx_input_ids)
        ctx_input_ids, ctx_attention_mask = verl_F.postprocess_data(
            input_ids=ctx_input_ids,
            attention_mask=ctx_attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=self.context_left_pad,
            truncation=self.truncation,
        )
        ctx_position_ids = compute_position_id_with_mask(ctx_attention_mask)

        resp_input_ids = response_ids_raw.unsqueeze(0)
        resp_attention_mask = torch.ones_like(resp_input_ids)
        resp_input_ids, resp_attention_mask = verl_F.postprocess_data(
            input_ids=resp_input_ids,
            attention_mask=resp_attention_mask,
            max_length=self.max_response_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=self.response_left_pad,
            truncation=self.truncation,
        )

        # full sequence for optional offline usage
        full_ids = torch.cat([context_ids_raw, response_ids_raw], dim=0).unsqueeze(0)
        full_attention_mask = torch.ones_like(full_ids)
        full_max_len = self.max_prompt_length + self.max_response_length
        full_ids, full_attention_mask = verl_F.postprocess_data(
            input_ids=full_ids,
            attention_mask=full_attention_mask,
            max_length=full_max_len,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        full_position_ids = compute_position_id_with_mask(full_attention_mask)

        response_mask = resp_attention_mask[0]

        row_dict_out = {
            "context_input_ids": ctx_input_ids[0],
            "context_attention_mask": ctx_attention_mask[0],
            "context_position_ids": ctx_position_ids[0],
            "response_ids": resp_input_ids[0],
            "response_attention_mask": resp_attention_mask[0],
            "response_mask": response_mask,
            "input_ids": full_ids[0],
            "attention_mask": full_attention_mask[0],
            "position_ids": full_position_ids[0],
            "agent_id": torch.tensor([agent_id], dtype=torch.long),
            "raw_prompt_ids": raw_prompt_ids,
        }

        if self.turn_id_key in row_dict:
            row_dict_out[self.turn_id_key] = int(row_dict[self.turn_id_key])
        if self.conversation_id_key in row_dict:
            row_dict_out[self.conversation_id_key] = row_dict[self.conversation_id_key]
        if "data_source" in row_dict:
            row_dict_out["data_source"] = row_dict.get("data_source")
        if "reward_model" in row_dict:
            row_dict_out["reward_model"] = row_dict.get("reward_model")
        if "extra_info" in row_dict:
            row_dict_out["extra_info"] = row_dict.get("extra_info")

        return row_dict_out
