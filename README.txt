


git clone https://github.com/sibmason/iris-classifier.git
cd iris-classifier
python -m venv venv && source venv\Scripts\activate
pip install -r requirements.txt
python src\train.py --test-size 0.2 --random-state 42


