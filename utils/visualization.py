import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             confusion_matrix)
from sklearn.preprocessing import label_binarize
import numpy as np
import seaborn as sns
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['svg.fonttype'] = 'none'

def plot_pr_curve(true_labels, pred_probs, task_name, class_dict, file_path=None):
    plt.figure(figsize=(5, 7))
    plt.rc("font", family='Times New Roman', size=12)
    from collections import Counter
    label_counts = Counter(true_labels)

    precision_recall_data = {}
    conf_matrix = None

    if len(pred_probs[0]) > 1:  # 多分类
        classes = range(len(pred_probs[0]))
        true_labels_bin = label_binarize(true_labels, classes=classes)
        conf_matrix = confusion_matrix(true_labels, np.argmax(pred_probs, axis=1))
        for class_idx in classes:
            precision, recall, thresholds = precision_recall_curve(
                true_labels_bin[:, class_idx],
                [probs[class_idx] for probs in pred_probs],
            )
            avg_precision = average_precision_score(
                true_labels_bin[:, class_idx], 
                [probs[class_idx] for probs in pred_probs]
            )
            class_name = class_dict.get(class_idx, f"Class {class_idx}")
            sample_count = label_counts.get(class_idx, 0)
            plt.plot(recall, precision, lw=2,
                     label=f'{class_name} (AP={avg_precision:.4f}, N={sample_count})')
            
            # 保存 precision 和 recall
            precision_recall_data[class_name] = {
                "precision": precision,
                "recall": recall,
                "thresholds": thresholds,
                "sample_count": sample_count,
                "avg_precision": avg_precision
            }
    else:  # 二分类
        precision, recall, thresholds = precision_recall_curve(true_labels, pred_probs)
        avg_precision = average_precision_score(true_labels, pred_probs)
        sample_count = len(true_labels)
        plt.plot(recall, precision, lw=2, label=f'Total (AP={avg_precision:.4f}, N={sample_count})')
        
        precision_recall_data["Total"] = {
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds,
            "sample_count": sample_count,
            "avg_precision": avg_precision
        }
        conf_matrix = confusion_matrix(true_labels, np.round(pred_probs))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    # plt.grid(alpha=0.3)

    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"PR curve saved to {file_path}")
    else:
        plt.show()
    plt.close()

    # 绘制混淆矩阵
    if conf_matrix is not None:
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_dict.values(), yticklabels=class_dict.values())
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'{task_name} Confusion Matrix')
        conf_matrix_path = file_path.replace('.svg', '_conf_matrix.svg') if file_path else None
        if conf_matrix_path:
            plt.savefig(conf_matrix_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {conf_matrix_path}")
        else:
            plt.show()
        plt.close()

    # 保存 precision 和 recall 到 txt 文件
    if file_path:
        txt_file = file_path.replace('.svg', '_metrics.txt')
        with open(txt_file, 'w') as f:
            for class_name, metrics in precision_recall_data.items():
                f.write(f'Class: {class_name}\n')
                f.write(f'Average Precision: {metrics["avg_precision"]:.4f}\n')
                f.write('Precision: ' + ', '.join(map(str, metrics['precision'])) + '\n')
                f.write('Recall: ' + ', '.join(map(str, metrics['recall'])) + '\n')
                f.write('Thresholds: ' + ', '.join(map(str, metrics['thresholds'])) + '\n')
                f.write(f'sample_count:  {metrics["sample_count"]}\n')
                f.write('\n')
        print(f"Precision and recall data saved to {txt_file}")



# precision_recall_data = {}
# with open('example_metrics.txt', 'r') as f:
#     lines = f.readlines()

# class_name = None
# for line in lines:
#     line = line.strip()
#     if line.startswith('Class:'):
#         class_name = line.split(': ')[1]
#         precision_recall_data[class_name] = {}
#     elif line.startswith('Average Precision:'):
#         precision_recall_data[class_name]['avg_precision'] = float(line.split(': ')[1])
#     elif line.startswith('Precision:'):
#         precision_recall_data[class_name]['precision'] = list(map(float, line.split(': ')[1].split(', ')))
#     elif line.startswith('Recall:'):
#         precision_recall_data[class_name]['recall'] = list(map(float, line.split(': ')[1].split(', ')))
#     elif line.startswith('Thresholds:'):
#         precision_recall_data[class_name]['thresholds'] = list(map(float, line.split(': ')[1].split(', ')))

# print(precision_recall_data)

