import numpy as np
import pandas as pd
import time
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import os

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 生成或加载数据（示例使用随机数据）
def load_data():
    # 示例：生成3个簇的二维数据
    np.random.seed(42)
    arr1 = np.random.randn(100, 2) + np.array([5, 5])
    arr2 = np.random.randn(100, 2) + np.array([-5, 5])
    arr3 = np.random.randn(100, 2) + np.array([0, -5])
    arr = np.vstack([arr1, arr2, arr3])
    
    # 实际应用中替换为你的数据
    # arr = np.load("your_data.npy")
    
    # 数据标准化
    scaler = StandardScaler()
    arr_scaled = scaler.fit_transform(arr)
    
    return arr, arr_scaled

# 聚类模型训练与评估
def train_and_evaluate_models(arr, arr_scaled, k_clusters=3, eps=0.5, min_samples=5):
    results = {}
    
    # 定义要比较的聚类方法
    clustering_methods = {
        "KMeans": KMeans(n_clusters=k_clusters, random_state=3),
        "GMM": GaussianMixture(n_components=k_clusters, random_state=3),
        "DBSCAN": DBSCAN(eps=eps, min_samples=min_samples),
        "SpectralClustering": SpectralClustering(n_clusters=k_clusters, random_state=3, affinity='nearest_neighbors')
    }
    
    # 创建保存结果的目录
    if not os.path.exists("clustering_results"):
        os.makedirs("clustering_results")
    
    # 对每种聚类方法进行训练和评估
    for name, model in clustering_methods.items():
        print(f"\n=== 正在训练 {name} 模型 ===")
        time_start = time.time()
        
        # 训练模型
        if name == "DBSCAN":  # DBSCAN没有单独的fit_predict方法
            labels = model.fit_predict(arr_scaled)
        elif name == "GMM":  # GMM使用predict方法
            model.fit(arr_scaled)
            labels = model.predict(arr_scaled)
        else:  # KMeans和SpectralClustering
            labels = model.fit_predict(arr_scaled)
        
        time_end = time.time()
        print(f"训练完成，耗时: {time_end - time_start:.2f}秒")
        
        # 计算评估指标
        try:
            # 确保至少有两个簇才能计算轮廓系数
            unique_labels = np.unique(labels)
            if len(unique_labels) <= 1:
                print(f"警告: {name} 只生成了一个簇，无法计算轮廓系数")
                silhouette = np.nan
            else:
                silhouette = silhouette_score(arr_scaled, labels)
            
            calinski = calinski_harabasz_score(arr_scaled, labels)
            davies = davies_bouldin_score(arr_scaled, labels)
            
            print(f"评估指标: 轮廓系数={silhouette:.4f}, Calinski-Harabasz指数={calinski:.4f}, Davies-Bouldin指数={davies:.4f}")
            
            # 保存结果
            results[name] = {
                "labels": labels,
                "silhouette": silhouette,
                "calinski": calinski,
                "davies": davies,
                "time": time_end - time_start,
                "model": model if name != "DBSCAN" else None  # DBSCAN没有保存模型的必要
            }
            
            # 保存模型（除了DBSCAN）
            if name != "DBSCAN":
                joblib.dump(model, f"clustering_results/{name}_model.pkl")
            
            # 可视化聚类结果
            plot_clusters(arr, labels, name, f"clustering_results/{name}_clusters.png")
            
        except Exception as e:
            print(f"Error: {e}")
            results[name] = None
    
    # 保存评估结果到CSV
    save_results_to_csv(results, "./clustering_results/comparison.csv")
    
    return results

# 可视化聚类结果
def plot_clusters(data, labels, method_name, save_path=None):
    plt.figure(figsize=(10, 8))
    
    # 获取唯一的标签
    unique_labels = np.unique(labels)
    
    # 为每个簇绘制点
    for label in unique_labels:
        if label == -1:  # DBSCAN中的噪声点
            plt.scatter(data[labels == label, 0], data[labels == label, 1], 
                        c='black', s=30, marker='x', label='Noise')
        else:
            plt.scatter(data[labels == label, 0], data[labels == label, 1], 
                        label=f'Clustering {label}', alpha=0.7)
    
    plt.title(f'{method_name} Clustering results')
    plt.xlabel('Feature-1')
    plt.ylabel('Feature-2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
        print(f"聚类结果图已保存至: {save_path}")
    
    plt.show()

# 保存评估结果到CSV
def save_results_to_csv(results, csv_path):
    # 准备数据框
    data = []
    for method, result in results.items():
        if result:
            data.append({
                '聚类方法': method,
                '轮廓系数': result['silhouette'],
                'Calinski-Harabasz指数': result['calinski'],
                'Davies-Bouldin指数': result['davies'],
                '训练时间(秒)': result['time']
            })
    
    if data:
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"评估结果已保存至: {csv_path}")
        
        # 打印结果表格
        print("\n=== 聚类方法性能比较 ===")
        print(df.to_string(index=False))

# 主函数
def main():
    print("=== 开始聚类方法比较 ===")
    
    # 加载数据
    arr, arr_scaled = load_data()
    print(f"数据形状: {arr.shape}")
    
    # 设置聚类参数
    k_clusters = 3  # KMeans、GMM和SpectralClustering的簇数量
    eps = 0.5       # DBSCAN的邻域半径
    min_samples = 5 # DBSCAN的最小样本数
    
    # 训练并评估模型
    results = train_and_evaluate_models(arr, arr_scaled, k_clusters, eps, min_samples)
    
    print("\n=== 聚类比较完成 ===")

if __name__ == "__main__":
    main()