* Clone Repo
    ```
    git clone --recursive https://dev.azure.com/nourteksolutions/AIML/_git/policy-classification-training-scripts
    ```

* System Requirements
    ```
    CUDA Version: 11.4
    1 x Tesla T4
    ```

* Create conda environment and install all libraries
    ```
    conda create -n classification_training python=3.9.16
    conda activate classification_training
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt
    ```

* Run process
    ```
    
    python step1_img_to_text.py # path data

    python step2_data_process.py

    python train.py

    tensorboard --logdir="logs/path"
    
    python inference


    ```