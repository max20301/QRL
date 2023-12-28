conda create --name QRL python=3.8
eval "$(conda shell.bash hook)"
conda activate QRL

pip install -r requirements.txt
pip install tfq-nightly
pip install gym[toy_text]

python quantum_reinforcement_learning.py

