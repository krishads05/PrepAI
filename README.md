# PrepAI - quick start (fully offline)
pip install -r requirements.txt
streamlit run app.py            # dashboard (recommended)

CLI alternative:
python run_layer1.py --syllabus sample_data/syllabus.txt --pyqs sample_data/pyqs --material sample_data/material --out outputs
python run_layer2.py prepare --material sample_data/material
python run_layer2.py quiz --n 8 --material sample_data/material
python run_layer2.py plan --minutes 120
Optional free local LLM: install Ollama, run `ollama pull qwen2.5:3b` - detected automatically.
