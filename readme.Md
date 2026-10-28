https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images

python 3.10.11

cd ai_detector_unimodel  
python -m venv venv  
venv\Scripts\activate
pip install -r requirements.txt
pip install pandas scikit-learn nltk joblib numpy
pip install language-tool-python
pip install pyspellchecker nltk
python build_hash_db.py
python app.py 




cd ai_detector_unimodel
venv\Scripts\activate
python app.py





