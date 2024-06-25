# multimodal_rag
download https://huggingface.co/google/siglip-so400m-patch14-384

download https://huggingface.co/openbmb/MiniCPM-V

change config/config.json for vis_model_path and model_LLM_path

conda create rag python==3.10

conda activate rag  (for latest anaconda/miniconda use source activate )

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

## run
python server.py
