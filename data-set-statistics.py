import data_factory as data
from analyze import analyze

for load in data.getAllDatasets():
    data, label = load()
    analyze(data, label)
