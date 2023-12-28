# QRL

## run code in google colab

[google colab]([https://www.runoob.com](https://colab.research.google.com/drive/15hjaBNVRBVmPhgYodLMNKSuRx8NEsTDq#scrollTo=yImT5bdoww6Y))

## run code through script

```bash
cd QRL
bash run.sh
```

## run code manually
1. first create and activate a virtual env through conda
    
    ```bash
    conda create --name QRL python=3.8
    conda activate QRL
    ```

2. install all packages through pip

    ```bash
    cd QRL
    pip install -r requirements.txt
    pip install tfq-nightly
    pip install gym[toy_text]
    ```

3. run our code

    ```bash
    python quantum_reinforcement_learning.py
    ```
