from datasets import load_dataset, list_datasets
from pprint import pprint

squad_train = load_dataset('squad_v2', split='train')
squad_valid = load_dataset('squad_v2', split='validation')
# print(squad_train)
pprint(squad_train[10008])