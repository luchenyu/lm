
# Universal LM Model

## Guidelines
1. Data is some kind of structure comprised of multiple "pieces" of different types and meaning.
2. Different "pieces" of one sample have inter-relationships with each other.
3. Task is about filling in the blank of different "pieces" of data.
4. There is no fundamental difference between "generation", "classification", "labeling", etc.
5. One universal model that can properly handle different types of "pieces" shall be able to solve any task.
6. Most part of the model should be task-independent and shared across tasks, and task-related parameters should be minimized.
7. Model should has zero-shot and few-shot learning ability, and survive from catastrophic forgetting when trained sequentially with different tasks.
7. For NLP tasks, the tokenize startegy should be flexible.
8. The model should be able to learn globally normalized matching score of context and token, instead of marginalized ones.

## Design
1. data_config
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
2. model_config
```python
{
    'char_embed_dim': int,
    'layer_size': int,
    'num_layers': int,
    'num_heads': int,
}
```
3. task_config
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
4. run_config
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
5. TransformerStruct
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
1. Model is word-based yet the basic unit is character. Model support flexible tokenization strategy. You can either limit the tokens(words) by feed a token vocab or go wild with unlimited vocabs.
2. Model has a speller module to generate tokens if no token vocab is provided.
3. Use transformer as encoder, each slot is defined by its field, position, and token.
4. Field embeds have control on the attention part. When model is freezed, we only train the field embeds.
5. Matcher takes the token embeds and token encodes of candidates to match, the later is for copy-mechanism.
6. Cross entropy training is performed globally, instead of sample-wise. So the logits can be directly used in beam decoding.
7. Dataset is independent of Task, one dataset can be mapped to multiple tasks.

## TODOs
1. support combine multiple dataset for same task
2. support multi-task training
3. try novel decoding, not from left to right
4. try better word embedder, maybe some clustering mechanism
5. learn word segmentation on the fly and optimize segmentor together
6. add support for more type of data, e.g. image and speech
7. explorer rl-based generation strategy, for data with many fields
