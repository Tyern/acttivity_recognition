conda env create -f environment.yml
pip install --user kaggle

PATH=$PATH:/home/tran/.local/bin
echo '{"username":"tyernd","key":"4b832cbf3b6dc58eb6cd38c5eb84469d"}' > ~/.kaggle/kaggle.json
chmod 600 /home/tran/.kaggle/kaggle.json

kaggle datasets download -p ./data tyernd/compgan-dataset
unzip data/compgan-dataset.zip