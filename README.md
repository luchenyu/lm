
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

## Design
1. data_config
```json
{
    'field_delim': str,
    'schema': [
        {
            'field_id': int,
            'group_id': int,
            'item_id': int,
            'type': 'sequence'|'class',
            'limited_vocab': bool,
            'token_vocab': None|path,
            'copy_from': [field_ids],
        },
    ],
}
```
2. model_config
```json
{
    'char_vocab_size': int,
    'char_vocab_dim': int,
    'char_vocab_emb': numpy.array,
    'layer_size': int,
    'num_layers': int,
    'num_heads': int
}
```
3. run_config
```json
{
    'batch_size': int,
    'max_train_steps': int,
    'max_lr': float,
    'pct_start': 0~1,
    'dropout': 0~1,
    'wd': float,
    'data': [
        {
            'target_level': int >= 0,
            'max_token_length': int,
            'min_seq_length': int,
            'max_seq_length': int
        }, //indexed by field_id, not default feature_id
    ],
}
```
4. features structure
```json
{
    'field_query_embeds': batch_size x length x num_layers*layer_size,
    'field_key_embeds': batch_size x length x num_layers*layer_size,
    'field_value_embeds': batch_size x length x num_layers*layer_size,
    'posit_embeds': batch_size x length x posit_size,
    'token_embeds': batch_size x length x embed_size,
    'masks': batch_size x length,
    'querys': batch_size x length x num_layers*layer_size,
    'keys': batch_size x length x num_layers*layer_size,
    'values': batch_size x length x num_layers*layer_size,
    'encodes': batch_size x length x layer_size
}
```

## Features
1. Model is word-based yet the basic unit is character. Model support flexible tokenization strategy. You can either limit the tokens(words) by feed a token vocab or go wild with unlimited vocabs.
2. Model has a speller module to generate tokens if no token vocab is provided.
3. Use transformer as encoder, each slot is defined by its field, position, and token.
4. Field embeds have control on the attention part. When model is freezed, we only train the field embeds.
5. Matcher takes the token embeds and token encodes of candidates to match, the later is for copy-mechanism.

## TODOs
1. implement reusable transformer
2. implement sequence decoding, start from simple way
3. implement beam search for sophisticated sequence generation
4. try better word embedder, maybe some clustering mechanism
5. learn word segmentation on the fly and optimize segmentor together
6. add support for more type of data, e.g. image and speech
7. explorer rl-based generation strategy, for data with many fields
