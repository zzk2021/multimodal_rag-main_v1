# multimodal_rag
download https://huggingface.co/google/siglip-so400m-patch14-384

download https://huggingface.co/openbmb/MiniCPM-V

change config/config.json

conda create rag python==3.10

conda activate rag  (for latest anaconda/miniconda use source activate )

pip install -r requirements.txt

## run
python server.py
