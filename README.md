
# Universal LM Model

## Guidelines
* Data is some kind of structure comprised of multiple "pieces" of different types and meaning.
* Different "pieces" of one sample have inter-relationships with each other.
* Task is about filling in the blank of different "pieces" of data.
* There is no fundamental difference between "generation", "classification", "labeling", etc.
* One universal model that can properly handle different types of "pieces" shall be able to solve any task.
* Most part of the model shall be task-independent and shared across tasks, while task-related parameters shall be minimized.
* Model shall have zero-shot and few-shot learning ability, and survive from catastrophic forgetting when trained sequentially with different tasks.
* For NLP tasks, the tokenize startegy shall be flexible.
* The model shall be able to learn globally normalized matching score of context and token, instead of marginalized ones.

## Design Specs
* data_config
```python
{
    'segment_delim': str,
    'data_index': [ ## list by segment_id
        {
            'field_id': int,
            'item_id': int,
        },
    ],
    'data_schema': [ ## list by field_id
        {
            'name': str,
            'type': 'sequence'|'class',
            'limited_vocab': bool,
            'token_vocab': None|path,
            'max_token_length': int,
            'min_seq_length': int,
            'max_seq_length': int,
            'group_id': int,
        },
    ],
}
```
* model_config
```python
{
    'char_embed_dim': int,
    'layer_size': int,
    'num_layers': int,
    'num_heads': int,
}
```
* task_config
```python
{
    'task_spec': [
        {
            'name': str,
            'type': 'sequence'|'class',
            'copy_from': [field_ids],
            'target_level': int >= 0,
            'group_id': int,
        },
    ],
}
```
* run_config
```python
{
    'task_config_name': str,
    'char_vocab_name': str,
    'model_config_name': str,
    'model_name': str,
    'warm_start_from': {
        'task_config_name': str,
        'model_name': str,
        'vars_to_warm_start': str,
    }
    'dataset_name': str,
    'field_mapping': [int,], ## list by field_id, list elem is field_id of dataset
    'hyper_params': {
        'batch_size': int,
        'max_train_steps': int,
        'max_lr': float,
        'pct_start': 0~1,
        'dropout': 0~1,
        'wd': float,
    },
}
```
* TransformerStruct
```python
{
    'field_query_embeds': tuple(batch_size x length x layer_size) * num_layers,
    'field_key_embeds': tuple(batch_size x length x layer_size) * num_layers,
    'field_value_embeds': tuple(batch_size x length x layer_size) * num_layers,
    'posit_embeds': batch_size x length x posit_size,
    'token_embeds': batch_size x length x embed_size,
    'masks': batch_size x length,
    'querys': tuple(batch_size x length x layer_size) * num_layers,
    'keys': tuple(batch_size x length x layer_size) * num_layers,
    'values': tuple(batch_size x length x layer_size) * num_layers,
    'encodes': batch_size x length x layer_size
}
```

## Features
* Model is word-based yet the basic unit is character. Model support flexible tokenization strategy. You can either limit the tokens(words) by feed a token vocab or go wild with unlimited vocabs.
* Model has a speller module to generate tokens if no token vocab is provided. (removed for speed)
* Use transformer as encoder, each slot is defined by its field, position, and token.
* Field embeds have control on the attention part. When model is freezed, we only train the field embeds.
* Matcher takes the token embeds and token encodes of candidates to match, the later is for copy-mechanism.
* Cross entropy training is performed globally, instead of sample-wise. So the logits can be directly used in beam decoding.
* Dataset is independent of Task, one dataset can be mapped to multiple tasks.
* Capable of multi-step generation/prediction, by setting target_level properly.

## TODOs
* Add support for combining multiple dataset of same task.
* Add support for multi-task training.
* Try novel decoding, not from left to right.
* Try better word embedder, maybe some clustering mechanism.
* Learn word segmentation on the fly and optimize segmentor together, unsupervisely if possible.
* Add support for more type of data, e.g. image and speech.
* Explorer rl-based generation strategy, for data with many fields.
* Automate the whole pipeline.
