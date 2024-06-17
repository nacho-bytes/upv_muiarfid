import json
import os
from sklearn.metrics import cohen_kappa_score

def load_folder_data(folder):
    annotations = []
    files = []
    for file_name in os.listdir(folder):
        if file_name.endswith(".json"):
            annotations.append(load_file_data(os.path.join(folder, file_name)))
            files.append(file_name)
    return annotations, files

def load_file_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_node_text(annotation, node_id):
    for node in annotation['AIF']['nodes']:
        if node['nodeID'] == node_id:
            return node['text']
    return None

def get_node_type(annotation, node_id):
    for node in annotation['AIF']['nodes']:
        if node['nodeID'] == node_id:
            return node['type']
    return None

def get_labels(annotation):
    labels = []
    for edgeIndex in range(len(annotation['AIF']['edges'])):
        if edgeIndex % 2 == 1:
            continue;
        start_node_text = get_node_text(annotation, annotation['AIF']['edges'][edgeIndex]['fromID'])
        relation_type = get_node_type(annotation, annotation['AIF']['edges'][edgeIndex]['toID'])
        end_node_text = get_node_text(annotation, annotation['AIF']['edges'][edgeIndex + 1]['toID'])
        labels.append((start_node_text + '_' + end_node_text, relation_type))
    return labels


def compare_annotators(y1_annotations, y2_annotations):
    y1_labels = get_labels(y1_annotations)
    y2_labels = get_labels(y2_annotations)

    labels = set()
    for label in y1_labels:
        labels.add(label[0])
    for label in y2_labels:
        labels.add(label[0])

    y1_vector = []
    y2_vector = []
    for label in labels:
        if label in [x[0] for x in y1_labels]:
            y1_vector.append([x[1] for x in y1_labels if x[0] == label][0])
        else:
            y1_vector.append('None')
        if label in [x[0] for x in y2_labels]:
            y2_vector.append([x[1] for x in y2_labels if x[0] == label][0])
        else:
            y2_vector.append('None')
    return cohen_kappa_score(y1_vector, y2_vector)

def main(folder):
    annotations, files = load_folder_data(folder)
    comparations = []
    for i in range(len(annotations)):
        for j in range(i + 1, len(annotations)):
            comparations.append(compare_annotators(annotations[i], annotations[j]))
            print(f'Comparing annotators {files[i]} and {files[j]}: {comparations[-1]}')
    print(f'Average comparation: {sum(comparations) / len(comparations)}')

if __name__ == "__main__":
    FOLDER = './analysis'
    main(FOLDER)
