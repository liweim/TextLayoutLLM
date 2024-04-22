from huggingface_hub import snapshot_download

snapshot_download(repo_id="meta-llama/llama-2-7b-hf",
                  local_dir='/dir/to/save',
                  local_dir_use_symlinks=False,
                  token='your token',
                  ignore_patterns=["*.pdf", "*.md", "*.txt"],
                  # allow_patterns=['tokenizer.model']
                  )
