# CrossSituationalLearning
This repository implements a dataset builder to create Cross Situational Learning dataset.

## Dependencies

You need numpy to run this package : `pip install numpy`.

## Usage

### Create the dataset

To create a Cross Situational Dataset, you need to create a `TwoSituationDataset` object and give it the following parameters:

- `objects` : list of string
- `colors` : list of string
- `positions` : list of string

You might want to use the same string for different categories. For example, you can use the same string for the color `orange` and the object `orange`.

You might want to use the same label for several different string in a categorie. For example, `middle` and `center` can be used to describe the same position, with only one unique label for both. In that case you can define those positions as follow: `positions = ['left', ('center', 'middle'), 'right']`. 

### Access the dataset

Once the TwoSituationDataset object is created, you can access the dataset as follow:
- `dataset.X` : numpy array of shape (n_sentences, n_words, n_categories)
- `dataset.Y` : numpy array of shape (n_sentences, n_words)


## Example

```python
from CSLDataset import TwoSituationDataset
import numpy as np

objects = ['glass', 'orange', 'cup', 'bowl']
colors = ['blue', 'orange', 'green', 'red']
positions = ['left', 'right', ('center', 'middle')]

dataset = TwoSituationDataset(objects=objects, colors=colors, positions=positions)
print(f'Shape of X: {dataset.X.shape}')
print(f'Shape of Y: {dataset.Y.shape}')
print()

random_indices = np.random.choice(len(dataset.sentences), 5, replace=False)
print("5 random sentences:")
for i in random_indices:
    print(dataset.sentences[i])
    print(dataset.predicates[i])
    print()
```
