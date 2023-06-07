from typing import Dict, Optional, Union, Tuple
import torch
from torch import nn
import torch.utils
import torch.utils.checkpoint
from dataclasses import dataclass
from transformers import SamModel
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    ModelOutput,
)
from transformers.models.sam.modeling_sam import (
    SAM_INPUTS_DOCSTRING,
    SamPatchEmbeddings,
    SamVisionLayer,
    SamVisionNeck,
)
from transformers.models.sam.configuration_sam import SamVisionConfig
from adaptor import PromptGenerator


@dataclass
class SamVisionEncoderOutput(ModelOutput):
    """
    Base class for sam vision model's outputs that also contains image embeddings obtained by applying the projection
    layer to the pooler_output.

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class SamImageSegmentationOutput(ModelOutput):
    """
    Base class for Segment-Anything model's output

    Args:
        iou_scores (`torch.FloatTensor` of shape `(batch_size, num_masks)`):
            The iou scores of the predicted masks.
        pred_masks (`torch.FloatTensor` of shape `(batch_size, num_masks, height, width)`):
            The predicted low resolutions masks. Needs to be post-processed by the processor
        vision_hidden_states  (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the vision model at the output of each layer plus the optional initial embedding outputs.
        vision_attentions  (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        mask_decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    iou_scores: Optional[torch.FloatTensor] = None
    pred_masks: Optional[torch.FloatTensor] = None
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vision_attentions: Optional[Tuple[torch.FloatTensor]] = None
    mask_decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


class SamVisionEncoderAdp(nn.Module):
    def __init__(self, config: SamVisionConfig):
        super().__init__()
        self.config = config
        self.image_size = config.image_size

        self.patch_embed = SamPatchEmbeddings(config)

        self.pos_embed = None
        if config.use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.parameter.Parameter(
                torch.zeros(
                    1,
                    config.image_size // config.patch_size,
                    config.image_size // config.patch_size,
                    config.hidden_size,
                )
            )

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            layer = SamVisionLayer(
                config,
                window_size=config.window_size
                if i not in config.global_attn_indexes
                else 0,
            )
            self.layers.append(layer)

        self.neck = SamVisionNeck(config)

        self.gradient_checkpointing = False
        self.prompt_generator = PromptGenerator(depth=config.num_hidden_layers)

    def get_input_embeddings(self) -> SamPatchEmbeddings:
        return self.patch_embed

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SamVisionEncoderOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.patch_embed(pixel_values)
        embedding_feature = self.prompt_generator.init_embeddings(
            hidden_states
        )
        handcrafted_feature = self.prompt_generator.init_handcrafted(
            pixel_values
        )
        prompt = self.prompt_generator.get_prompt(
            handcrafted_feature, embedding_feature
        )
        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed

        all_hidden_states: Optional[Tuple[torch.FloatTensor]] = (
            tuple() if output_hidden_states else None
        )
        all_self_attentions: Optional[Tuple[torch.FloatTensor]] = (
            tuple() if output_attentions else None
        )
        B, H, W = hidden_states.shape[0:3]
        for idx, layer_module in enumerate(self.layers):
            if output_hidden_states and all_hidden_states is not None:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = prompt[idx].reshape(B, H, W, -1) + hidden_states
            layer_outputs = layer_module(
                hidden_states, output_attentions=output_attentions
            )

            hidden_states = layer_outputs[0]

            if output_attentions and all_self_attentions is not None:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states and all_hidden_states is not None:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.neck(hidden_states)

        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_self_attentions,)
            return outputs

        return SamVisionEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class SamPT(SamModel):
    """Modify SamModel to realize prompt tuning

    __init__: add prompt_embedding.
    forward: utilize prompt_embedding, in the both sparse and dense embedding.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.hidden_size = 256
        # [pt_length, hidden_size] -> \
        #           [bt_size, point_batch_size, nb_points_per_image, hidden_size]
        self.sparse_prompt = nn.Embedding(20, self.hidden_size)

    @add_start_docstrings_to_model_forward(SAM_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
        image_embeddings: Optional[torch.FloatTensor] = None,
        multimask_output: bool = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict=None,
        **kwargs,
    ) -> Union[Tuple, Dict[str, torch.Tensor]]:
        r"""
        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoModel, AutoProcessor

        >>> model = AutoModel.from_pretrained("facebook/sam-vit-base")
        >>> processor = AutoProcessor.from_pretrained("facebook/sam-vit-base")

        >>> img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-car.png"
        >>> raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        >>> input_points = [[[400, 650]]]  # 2D location of a window on the car
        >>> inputs = processor(images=raw_image, input_points=input_points, return_tensors="pt")

        >>> # Get segmentation mask
        >>> outputs = model(**inputs)

        >>> # Postprocess masks
        >>> masks = processor.post_process_masks(
        ...     outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
        ... )
        ```
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        if pixel_values is None and image_embeddings is None:
            raise ValueError(
                "Either pixel_values or image_embeddings must be provided."
            )

        if pixel_values is not None and image_embeddings is not None:
            raise ValueError(
                "Only one of pixel_values and image_embeddings can be provided."
            )

        if input_points is not None and len(input_points.shape) != 4:
            raise ValueError(
                "The input_points must be a 4D tensor. Of shape `batch_size`, `point_batch_size`, `nb_points_per_image`, `2`.",
                " got {}.".format(input_points.shape),
            )
        if input_boxes is not None and len(input_boxes.shape) != 3:
            raise ValueError(
                "The input_points must be a 3D tensor. Of shape `batch_size`, `nb_boxes`, `4`.",
                " got {}.".format(input_boxes.shape),
            )
        if input_points is not None and input_boxes is not None:
            point_batch_size = input_points.shape[1]
            box_batch_size = input_boxes.shape[1]
            if point_batch_size != box_batch_size:
                raise ValueError(
                    "You should provide as many bounding boxes as input points per box. Got {} and {}.".format(
                        point_batch_size, box_batch_size
                    )
                )

        image_positional_embeddings = (
            self.get_image_wide_positional_embeddings()
        )
        # repeat with batch size
        batch_size: Optional[int] = None
        if pixel_values is not None:
            batch_size = pixel_values.shape[0]
        elif image_embeddings is not None:
            batch_size = image_embeddings.shape[0]
        else:
            raise ValueError(
                "Either pixel_values or image_embeddings must be provided."
            )

        image_positional_embeddings = image_positional_embeddings.repeat(
            batch_size, 1, 1, 1
        )

        vision_attentions = None
        mask_decoder_attentions = None
        vision_hidden_states = None

        if pixel_values is not None:
            vision_outputs = self.vision_encoder(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            image_embeddings = vision_outputs[0]

            if output_hidden_states:
                vision_hidden_states = vision_outputs[1]
            if output_attentions:
                vision_attentions = vision_outputs[-1]

        if input_points is not None and input_labels is None:
            input_labels = torch.LongTensor(
                torch.ones_like(
                    input_points[:, :, :, 0], device=input_points.device
                )
            )

        if image_embeddings is not None:
            if (
                input_points is not None
                and image_embeddings.shape[0] != input_points.shape[0]
            ):
                raise ValueError(
                    "The batch size of the image embeddings and the input points must be the same. ",
                    "Got {} and {} respectively.".format(
                        image_embeddings.shape[0], input_points.shape[0]
                    ),
                    " if you want to pass multiple points for the same image, make sure that you passed ",
                    " input_points of shape (batch_size, point_batch_size, num_points_per_image, 3) and input_labels of shape (batch_size, point_batch_size, num_points_per_image)",
                )

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )

        "add prompts"
        prompts = self.sparse_prompt.weight[None, None, :, :]
        prompts = prompts.repeat(batch_size, 1, 1, 1)
        sparse_embeddings = torch.cat((sparse_embeddings, prompts), dim=-2)

        (
            low_res_masks,
            iou_predictions,
            mask_decoder_attentions,
        ) = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            output_attentions=output_attentions,
        )

        if not return_dict:
            output = (iou_predictions, low_res_masks)
            if output_hidden_states:
                output = output + (vision_hidden_states,)

            if output_attentions:
                output = output + (vision_attentions, mask_decoder_attentions)
            return output

        return SamImageSegmentationOutput(
            iou_scores=iou_predictions,
            pred_masks=low_res_masks,
            vision_hidden_states=vision_hidden_states,
            vision_attentions=vision_attentions,
            mask_decoder_attentions=mask_decoder_attentions,
        )


class SamAdp(SamPT):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.vision_encoder = SamVisionEncoderAdp(config.vision_config)


def print_class_hierarchy(cls, indent=0) -> None:
    """print the type tree of a instance

    Args:
        instance.__class__
    """
    print(" " * indent + cls.__name__)
    for base_cls in cls.__bases__:
        print_class_hierarchy(base_cls, indent + 4)


if __name__ == "__main__":
    test_output = SamImageSegmentationOutput()
