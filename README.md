
# Universal LM Model

## Guidelines:
1. data is structure comprised of multiple "pieces" of different type and meaning
2. "pieces" of one sample have inter-relationships with each other
3. different task is merely about filling the blank in different data structures
4. there is no difference in "generation", "classification", "labeling", etc., they are all the same
5. we can solve all the tasks using one universal model, which can properly handle different types of "pieces"
6. we should reuse as much as possible for the parameters of the universal model

## TODOs
1. implement reusable transformer
2. implement sequence decoding, start from simple way
3. implement beam search for sophisticated sequence generation
4. copy mechanism support, currently there is no explicit copy
5. try better word embedder, maybe some clustering mechanism
6. learn word segmentation on the fly and optimize segmentor together
7. add support for more type of data, e.g. image and speech
8. explorer rl-based generation strategy, for data with many fields
