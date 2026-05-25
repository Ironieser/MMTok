# MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs
# Paper: https://arxiv.org/abs/2508.18264
# This file is modified from the official Qwen2.5-VL code (https://github.com/QwenLM/Qwen2.5-VL).
# Copyright (c) 2025 Zoom Communications, Inc. Author: Sixun Dong.

"""
MMTok Qwen2.5-VL model: overridden forward with token selection (image -> Vision Tower -> MMTok select -> Qwen2.5-VL).
"""

import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from loguru import logger as eval_logger
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModelOutputWithPast, auto_docstring


class Qwen2_5_VL_MMTok(nn.Module):
    """
    MMTok Qwen2.5-VL: forward runs vision -> MMTok selection -> language model.
    """

    def get_video_features(self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        video_embeds, image_features = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        return video_embeds, image_features

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds, image_features = self.visual(pixel_values, grid_thw=image_grid_thw)
        return image_embeds, image_features

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLModelOutputWithPast]:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size)):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Qwen2_5_VLImageProcessor.__call__`] for details. [`Qwen2_5_VLProcessor`] uses
            [`Qwen2_5_VLImageProcessor`] for processing videos.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        select_pixel = False
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                select_pixel = True
                image_embeds, image_features = self.get_image_features(pixel_values, image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}")

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}")

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None or (past_key_values is None or past_key_values.get_seq_length() == 0):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if select_pixel:
            question = self.get_question()
            try:
                from src.ttrv_pruning.mmtok_ttrv import question_from_context

                question = question_from_context(self) or question
            except ModuleNotFoundError:
                pass
            text_token_embedding = None
            if input_ids is not None and inputs_embeds is not None:
                text_mask = input_ids[0] != self.config.image_token_id
                video_token_id = getattr(self.config, "video_token_id", None)
                if video_token_id is not None:
                    text_mask = text_mask & (input_ids[0] != video_token_id)
                if attention_mask is not None and attention_mask.ndim == 2:
                    text_mask = text_mask & attention_mask[0].to(device=text_mask.device, dtype=torch.bool)
                if text_mask.any():
                    text_token_embedding = inputs_embeds[0, text_mask].detach()
            token_retain_ratio = getattr(self._mmtok_core, "retain_ratio", float(os.environ.get("TOKEN_RETAIN_RATIO", "0.1")))
            target_vision_tokens = int(token_retain_ratio * image_embeds.shape[0])
            selection_method = os.environ.get("SELECTION_METHOD", "mmtok").lower()

            cached_indices = None
            if selection_method == "mmtok":
                try:
                    from src.ttrv_pruning.mmtok_ttrv import cached_mmtok_selection

                    cached_indices = cached_mmtok_selection(
                        self,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        num_visual_tokens=int(image_embeds.shape[0]),
                        row_index=0,
                    )
                except ModuleNotFoundError:
                    cached_indices = None

            if selection_method == "mmtok":
                if cached_indices is not None:
                    selected_indices = cached_indices
                    selected_tensor = torch.as_tensor(selected_indices, dtype=torch.long, device=image_embeds.device)
                    selected_image_embeds = image_embeds[selected_tensor]
                    del selected_tensor
                else:
                    selected_indices, selected_image_embeds = self._mmtok_core.apply_selection_preprocess_qwen(
                        image_embeds,
                        image_features,
                        question,
                        target_vision_tokens=target_vision_tokens,
                        text_token_embedding=text_token_embedding,
                    )
                    try:
                        from src.ttrv_pruning.mmtok_ttrv import remember_mmtok_selection

                        remember_mmtok_selection(
                            self,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            num_visual_tokens=int(image_embeds.shape[0]),
                            selected_indices=selected_indices,
                            row_index=0,
                        )
                    except ModuleNotFoundError:
                        pass
            elif selection_method == "divprune":
                def DivPrune(visual_feature_vectors, target_vision_tokens=1):
                    # threshold_terms = int(round(threshold_ratio*image_feature_length))
                    threshold_terms = target_vision_tokens

                    def pairwise_cosine_similarity(matrix):
                        norm_matrix = matrix / matrix.norm(dim=1, keepdim=True)
                        cosine_similarity = torch.mm(norm_matrix, norm_matrix.t())
                        return cosine_similarity

                    cosine_matrix = 1.0 - (pairwise_cosine_similarity(visual_feature_vectors))

                    s = torch.empty(threshold_terms, dtype=torch.long, device=visual_feature_vectors.device)
                    selected_mask = torch.zeros(cosine_matrix.size(0), dtype=torch.bool, device=cosine_matrix.device)
                    for i in range(threshold_terms):
                        if i == 0:
                            m2 = cosine_matrix
                        else:
                            m2 = torch.index_select(cosine_matrix, 0, torch.index_select(s, 0, torch.arange(0, i, device=cosine_matrix.device)))

                        if i == 0:
                            scores = torch.topk(m2, 2, dim=0, largest=False).values[1, :]  # for distance
                        else:
                            scores = torch.min(m2, dim=0).values  # for distance

                        scores[selected_mask] = float("-inf")

                        phrase_to_add_idx = torch.argmax(scores)

                        s[i] = phrase_to_add_idx
                        selected_mask[phrase_to_add_idx] = True

                    s.sort()
                    return s

                selected_indices = DivPrune(image_features, target_vision_tokens=target_vision_tokens)
                selected_image_embeds = image_embeds[selected_indices]
            else:
                raise ValueError(f"[MMTok] Unsupported SELECTION_METHOD: {selection_method}, use 'mmtok' or 'divprune'")

            select_mask = torch.zeros(image_embeds.shape[0], dtype=torch.bool, device=image_embeds.device)
            select_mask[selected_indices] = True
            keep_sequence = None
            img_mask = (input_ids == self.config.image_token_id)[0]
            st_idx = torch.nonzero(img_mask, as_tuple=True)[0]

            if st_idx.numel() > 0:
                first, last = st_idx[0].item(), st_idx[-1].item()
                if len(selected_indices) == 0:
                    # Remove all vision tokens, keep text only
                    keep_sequence = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
                    keep_sequence[first : last + 1] = False
                    position_ids = position_ids[:, :, keep_sequence]
                    if attention_mask is not None:
                        attention_mask = attention_mask[:, keep_sequence]
                    inputs_embeds = inputs_embeds[:, keep_sequence]
                else:
                    img_mask[first : last + 1] = ~select_mask
                    img_mask = ~img_mask
                    keep_sequence = img_mask.detach().clone()
                    selected_positions = first + torch.as_tensor(
                        selected_indices, dtype=torch.long, device=img_mask.device
                    )
                    inputs_embeds[:, selected_positions] = selected_image_embeds

                    position_ids = position_ids[:, :, keep_sequence]
                    if attention_mask is not None:
                        attention_mask = attention_mask[:, keep_sequence]
                    inputs_embeds = inputs_embeds[:, keep_sequence]
                    del selected_positions
            try:
                from src.ttrv_pruning.mmtok_ttrv import record_mmtok_stage

                record_mmtok_stage(
                    self,
                    selected_indices=selected_indices,
                    num_visual_tokens=int(image_embeds.shape[0]),
                    row_index=0,
                    keep_sequence=keep_sequence,
                )
            except ModuleNotFoundError:
                pass
            del image_embeds, image_features, selected_indices, selected_image_embeds
            del select_mask, img_mask, st_idx, keep_sequence
            del question, text_token_embedding, token_retain_ratio, target_vision_tokens, selection_method, cached_indices
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )

        output = Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()
