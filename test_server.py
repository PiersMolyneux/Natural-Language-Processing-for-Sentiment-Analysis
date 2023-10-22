import requests

print('Base')
print(requests.get('http://0.0.0.0:8000/').json())
print()
print('Testing review 1')
review = {'text': 'This film was absolutely dreadful. Terrible. I am never going to watch it again. Horrible.'}
response = requests.post('http://0.0.0.0:8000/predict/', json=review)
print(response.json())

print('Testing review 2')
review = {'text': 'This film was incredible. Must watch. Go to cinemas now. The acting was amazing. I cannot wait for the sequel'}
response = requests.post('http://0.0.0.0:8000/predict/', json=review)
print(response.json())


