
# Universal LM Model

## Guidelines:
1. Data is some kind of structure comprised of multiple "pieces" of different types and meaning.
2. Different "pieces" of one sample have inter-relationships with each other.
3. Task is about filling in the blank of different "pieces" of data.
4. There is no fundamental difference between "generation", "classification", "labeling", etc.
5. One universal model that can properly handle different types of "pieces" shall be able to solve any task.
6. Most part of the model should be task-independent and shared across tasks, and task-related parameters should be minimized.
7. Model should has zero-shot and few-shot learning ability, and survive from catastrophic forgetting when trained sequentially with different tasks.
7. For NLP tasks, the tokenize startegy should be flexible.

## TODOs
1. implement reusable transformer
2. implement sequence decoding, start from simple way
3. implement beam search for sophisticated sequence generation
4. copy mechanism support, currently there is no explicit copy
5. try better word embedder, maybe some clustering mechanism
6. learn word segmentation on the fly and optimize segmentor together
7. add support for more type of data, e.g. image and speech
8. explorer rl-based generation strategy, for data with many fields
