from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import homogeneity_completeness_v_measure
import pandas as pd

# generate result for Classification phase
print(10*"-" + " Classification Result " + 10*"-")
datasetClassified = pd.read_csv('./data/datasetClassified.csv', sep = ',', header = 0)
classes_ = ["F", "NF"]
acc = accuracy_score(datasetClassified['label'], datasetClassified['predClass'])
print(f"accuracy of the classifier is {acc*100} %")
cm = confusion_matrix(datasetClassified['label'], datasetClassified['predClass'], labels = classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = classes_)
disp.plot()
plt.show()


print(10*"-" + " Topic Modeling Result " + 10*"-")
datasetClustered = pd.read_csv('./data/datasetClustered.csv', sep = ',', header = 0)

labels_true = datasetClustered['ProjectID']
labels_pred = datasetClustered['Topic']

h, c, v = homogeneity_completeness_v_measure(labels_true, labels_pred)

print('Homogeneity:', h)
print('Completeness:', c)
print('V-measure:', v)
