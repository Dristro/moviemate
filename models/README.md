# Models

All model implementations here. This is where all the models
run. Other modules in the project can call a model from here.

Following attrs are availible:
* embed(str)
* generate_from_embed(Tensor)
* decode(Tensor)
* encode(str | list[str])
* generate_from_str(str)
* write_query(Tensor)


## Model details:

* Embedding Model: `all-mpnet-base-v2`
* Chat Model: `Qwen2.5-7B-Instruct`
* Code Model: `Qwen2.5-7B-Instruct`
* Enriching Model: `Qwen2.5-1.5B-Instruct`



|Model name|Embedding dim|Max context size|Vocab size|Model purpose|Model config file|
|-|-|-|-|-|-|
|`all-mpnet-base-v2`|768|514|30527|Embedding Model|[huggingface](https://huggingface.co/sentence-transformers/all-mpnet-base-v2/blob/main/config.json)|
|`Qwen2.5-7B-Instruct`|3584|32768|152064|Chat/Code Model|[huggingface](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)|
|`Qwen2.5-1.5B-Instruct`|1536|32768|151936|Enriching Model|[huggingface](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json)|

All model information is 'sourced' from their `config.json` files on HF.
